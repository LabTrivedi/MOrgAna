# README for transform_into_stacks_v6

Setup: Download the transform_into_stacks.ijm 
Open Fiji -> Plugins -> Macros -> Run..
First Window -> select the transform_into_stacks_v6.ijm 
Second Window -> select the folder of tif images

INPUT: folder of tif images of different channels and different conditions

Images when sorted in the folder should have the following order:
A1 - channel1
A1 - channel2
A1 - channel3
A2 - channel1
A2 - channel2
A2 - channel3
.
.
.

Click Run.
1) Select directory of folder containing all tifs
2) Select number of channels (including Brightfield) and indicate if images are of a fully imaged 96-well plate.
3) Select for each channel (Brightfield is by default the first channel, do not change this):
=> give a name for the channel 
=> starting image/ first image in that channel in folder
=> increment => when the next image is in that channel (if in doubt, leave option as default)
(identical inputs to the Image Sequence option in IJ)

ex. in example above, 
channel 1 -> start = 1; increment = 3;
channel 2 -> start = 2; increment = 3;
channel 3 -> start = 3; increment = 3;

4) BF assumed to be in gray LUT -> choose LUT for all other channels
5) if 96-well plate: Choose number of conditions (wells are split by columns) and your names for your conditions in the respective columns.
6) Wait. When completed, in Fiji Log: Completed! Check out the output!


Output: 
If fully imaged 96-well plate:
1) one folder for each condition, with images named by its well name. (ex. 1A, 1B, 1C)
2) one overview folder with a tif montage for each condition with all samples in all channels.
Otherwise
1) one folder for images named by its order (ex. 000, 001, 002...)
2) one overview folder with a tif montage for all samples in all channels.


Hidden features:
1) Greyscale moved to the front, will always be the first channel
