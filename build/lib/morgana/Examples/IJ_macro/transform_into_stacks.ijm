// This plugin creates stacks for each well (with channels merged in one stack)
// Auto threshold by Otsu method applied across each condition 
//
// Input: folder of tif images, with alternating channels ex. BF, GFP, RFP, BF, GFP, RFP, etc.
// Input can contain only one channel; assumed to be brightfield
// Input is assumed to be imaged in horizontal order (by rows, not columns)
// Output: one folder for each condition in case of fully imaged 96-well plate or one folder for others
// plus one folder for overview images, montage tif of all images in the different channels
//
// Fluorescent channels cannot have the same LUT or error will occur
//
// Author: Jia Le Lim
//
// Please email JL in case of errors or questions
// Last edited on 05/05/2021

// 27/03/2020 Fixed bug in excluded stacks

// Naming of stacks to be by a three digit number in case of non-96-well plate
function addingzeros(number) { 
        number = toString(number);
        for (len = lengthOf(number); len < 3; len++){number = "0" + number;};
        return number; 
};


// User chooses input folder and an output folder is created in choosen directory
path = getDirectory("Choose Directory of folder"); 
result_path = path + "output"; excluded_path = result_path + "/overviews"
File.makeDirectory(result_path); File.makeDirectory(excluded_path); 
print("Output found in: " + result_path); print("Please wait...");

////////////////////////////////////////////////// ALL USER INPUTS
// Select number of channels, indicate if full 96 wells imaged
Dialog.create("Imaging plate");
Dialog.addNumber("Number of Channels", 2); 
Dialog.addCheckbox("Images of all 96 wells of a 96-well plate?", true); 
Dialog.show(); channel_nr = Dialog.getNumber(); is_96wellplate = Dialog.getCheckbox();

// Select number of conditions if 96wellplate is fully imaged
// Nr. of columns for each condition = 12/condition_nr
if (is_96wellplate){
	Dialog.create("Number of conditions");
	condition_choices = newArray("1", "2", "3", "4", "6", "12");
	Dialog.addChoice("Number of conditions", condition_choices);
	Dialog.show(); 
	condition_nr = parseInt(Dialog.getChoice());
	
	// Get names of conditions
	Dialog.create("Names of conditions on 96-well plate");
	mock_conditions = newArray("control", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K");
	for (i = 0; i < condition_nr ; i++){
		start_col = (1 + (12 / condition_nr) * i); end_col = (12 / condition_nr) * (i + 1);
		label = "Condition " + toString(i) + ": Col " + toString(start_col) + " - " + toString(end_col);
		Dialog.addString(label, mock_conditions[i]);
	}; Dialog.show();
	cond_names = newArray(condition_nr); 
	for (i = 0; i < condition_nr; i++){cond_names[i] = Dialog.getString(); };
	
	// Creating the names of wells in 96-well plate [1A, 2A, 3A, ... 11H, 12H]
	well_names = newArray(96);
	plate_row = newArray("A", "B", "C", "D", "E", "F", "G" ,"H");
	plate_index = 0;
	for (j = 0; j < 8; j++) {
		for (col = 1; col < 13; col++) { // columns of plate 1-12
			well_names[plate_index] = toString(col) + plate_row[j];
			plate_index++;
	};};
	
	// Creates list of numbers for stack creation for each condition & creates directories for each condition
	colspercond = 12/condition_nr; columns = newArray(condition_nr); conds_path = newArray(condition_nr);
	for (i = 0; i < condition_nr; i++){ // i = condition_nr
		start_nr = i * colspercond + 1; // first nr in each condition
		columns[i] = toString(start_nr);
		conds_path[i] = result_path + "/" + cond_names[i]; File.makeDirectory(conds_path[i]);
		for (e = 1; e < colspercond; e++){ // go horizontally across first row
			columns[i] = columns[i] + ","; columns[i] = columns[i] + toString(start_nr + e); };
		for (z = 1; z < 8; z++){ // go vertically down plate
			for (j = 0; j < colspercond; j++){ // go horizontally across plate
				row_nr = start_nr + j + z * 12;
				if (row_nr < 97){ // there are only 96 wells
					columns[i] = columns[i] + ","; columns[i] = columns[i] + toString(row_nr); };
	};};};
}; else { // for cases where is_96wellplate is false
	condition_nr = 1; cond_names = newArray(condition_nr); cond_names[0] = "condition1"; 
	columns = newArray(condition_nr); columns[0] = "1";
	conds_path = newArray(condition_nr); 
	conds_path[0] = result_path + "/" + cond_names[0]; File.makeDirectory(conds_path[0]);
};

// Get image sequence parameters (same input as for image sequence tool in FIJI)
// -> start is the first image for each channel in folder when sorted
// -> increment is every subsequent image found for each channel
// when in doubt for increment, leave the default value 
if (channel_nr >1){
	Dialog.create("Select Image Sequence for each channel");
	Dialog.addMessage("<html><u><b>Brightfield</b></u>"); 
	Dialog.addString("Name of BF channel: ", "BrightField");
	Dialog.addString("Starting", "1"); Dialog.addString("Increment", toString(channel_nr));
	for (i = 1; i < channel_nr; i++) {
		Dialog.addMessage("<html><u><b>Fluorescent channel " + i + "</b></u>"); 
		Dialog.addString("Name of fluorescent channel: ", "GFP");
		Dialog.addString("Starting", "2"); Dialog.addString("Increment", toString(channel_nr));
	}; Dialog.show();
	// Store user inputs for image sequences [channel0/BF, channel1, channel2, ...]
	channel_names = newArray(channel_nr);
	start_nrs = newArray(channel_nr); increment_nrs = newArray(channel_nr);
	for (i = 0; i < channel_nr; i++){
		channel_names[i] = Dialog.getString();
		start_nrs[i] = Dialog.getString(); increment_nrs[i] = Dialog.getString();
	};
	// Choose the LUT for each channel, Assumes that the first channel is Brightfield -> C4
	Dialog.create("Select LUT for each channel");
	LUT_choices = newArray("red", "green", "blue", "gray", "cyan", "magenta", "yellow");
	for (i = 1; i < channel_nr; i++){
		Dialog.addChoice(channel_names[i], LUT_choices);
	}; Dialog.show(); 
	chosen_LUT = newArray(channel_nr); chosen_LUT[0] = 4;
	for (i = 1; i < channel_nr; i++){
		userLUT = Dialog.getChoice();
		for (j = 0; j < lengthOf(LUT_choices); j++){ // return index of LUT
			if ( userLUT == LUT_choices[j]){ chosen_LUT[i] = j+1;}; 
	};};
}; else {
	channel_names = newArray(channel_nr); start_nrs = newArray(channel_nr); 
	increment_nrs = newArray(channel_nr);
	channel_names[0] = "BrightField"; start_nrs[0] = "1"; increment_nrs[0] = "1";
}; 

////////////////////////////////////////////////// Starts compiling
setBatchMode("hide");

// Creates the stack for each channel and each condition
is_settings = " sort  file=tif"; montage_settings = " scale=0.25 border=0 font=28 label";
for (i = 0; i < channel_nr; i++){
	imageseq = "open='" + path + "' starting=" + start_nrs[i] + " increment=" + increment_nrs[i] + is_settings;
	run("Image Sequence...", imageseq); // Opens the images you require
	for (j = 0; j < condition_nr; j++){
		if (is_96wellplate){
			substackstr = "  slices=" + columns[j]; run("Make Substack...", substackstr);
		};
		filename = "Substack_" + cond_names[j] + "_" + channel_names[i] + ".tif";
		setAutoThreshold("Otsu dark stack"); // For nice visual preliminary view
		saveasstr = conds_path[j] + "/" + filename; saveAs("Tiff", saveasstr); 
		close(); // close substack
		};
}; run("Close All"); // close all opened image seq

// create stack for each condition
for (i = 0; i < condition_nr; i++){ // for each condition
	merging_str = ""; otherchannel_placement = ""; count = 1;
	if (is_96wellplate){
		well_nrs = split(columns[i], ","); // well number to match corresponding well names
	}; else { well_nrs = newArray(1);};
	for (j = 0; j < channel_nr; j++){ // Open all stacks for one condition
		stack_name = "Substack_" + cond_names[i] + "_" + channel_names[j] + ".tif";
		stack_path = conds_path[i] + "/" + stack_name; open(stack_path);
		if (channel_nr > 1){ merging_str = merging_str + "c" + toString(chosen_LUT[j]) + "=" + stack_name + " ";}
	};  
	if (channel_nr > 1){ 
		merging_str = merging_str + "create"; run("Merge Channels...", merging_str); // Merge channels
		// move Brightfield as the first channel
		order_LUT = Array.copy(chosen_LUT); Array.sort(order_LUT); 
		for (h = 0; h < lengthOf(order_LUT); h++){ // return placement of BF channel
			if (order_LUT[h] == 4){ BF_placement = h+1;}; 
			else {otherchannel_placement = otherchannel_placement + toString(h+1); }; };
		if (channel_nr > 1){
			arrange_seq = "new=" + toString(BF_placement) + otherchannel_placement;
			run("Arrange Channels...", arrange_seq); };
	};
	for(slice = 1; slice <= nSlices; slice++){ // rename slices with names of wells
		if (is_96wellplate){
			setSlice(slice); setMetadata("Label", well_names[parseInt(well_nrs[floor((slice-1)/channel_nr)])-1]);
		}; else {
			setSlice(slice); setMetadata("Label", "" + addingzeros(floor((count-1)/channel_nr))); count++;};
	};
	//// Make overview
	if ((count >= 61)|(lengthOf(well_nrs)>=61)){col_nr = 8;}; else {col_nr = 6;};
	example_info_str = "columns=" + toString(col_nr) + " scale=0.25 border=2 font=28 label";	
	run("Make Montage...", example_info_str); 
	example_path = excluded_path + "/overview_" + cond_names[i] + ".tif";
	saveAs("tif", example_path); close();

	// Save the main stack 
	final_stack_name = conds_path[i] + "/Stack_" + cond_names[i] + ".tif"; saveAs("Tiff", final_stack_name); 
			
run("Close");
}; run("Close All"); // Please close ALL windows here before next step

// Create stacks for each well from the hyperstacks
for (i = 0; i < condition_nr; i++){
	stack_path = conds_path[i] + "/" + "Stack_" + cond_names[i] + ".tif"; 
	open(stack_path);
	slice = 1; // note that nSlices = number of wells * channel_nr 
	while (slice <= nSlices){ // rename for example labelling
		setSlice(slice); well_label = getInfo("slice.label"); 
		ss_str = "channels=1-" + toString(channel_nr) + " slices=" + toString(((slice-1)/channel_nr)+1);
		run("Make Substack...", ss_str); // note that slices = actual slice nr in stack
		well_stack_name = conds_path[i] + "/" + well_label + ".tif";
		saveAs("Tiff", well_stack_name); close(); slice = slice + channel_nr; 
}; close();
}; run("Close All");

// delete hyperstacks
for (i = 0; i < condition_nr; i++){
	stack_path = conds_path[i] + "/" + "Stack_" + cond_names[i] + ".tif"; 
	ok = File.delete(stack_path);
}; 

// Delete other files
for (i = 0; i < condition_nr; i++){ // Delete substacks
	for (j = 0; j < channel_nr; j++){ 
		to_delete = conds_path[i] + "/Substack_" + cond_names[i] + "_" + channel_names[j] + ".tif"; 
		ok = File.delete(to_delete); 
}; };

print("Completed! Check out the output! ");

	