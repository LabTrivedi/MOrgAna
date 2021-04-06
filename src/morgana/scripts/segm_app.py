#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:50 2019

@author: ngritti
"""
import sys, os
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filepath,'..'))
import PyQt5.QtWidgets
import GUIs.mainwindow

if __name__ == '__main__':
    def run():
        app = PyQt5.QtWidgets.QApplication(sys.argv)
        gallery = GUIs.mainwindow.gastrSegmentApp()
        gallery.show()
        sys.exit(app.exec_())

    run()

