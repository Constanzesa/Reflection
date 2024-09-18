#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Wed Sep 18 01:55:58 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'collection_stud'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'sex': '',
    'age': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/arnavkapur/Desktop/MIT/Conflict - Detection/Psychopy/Collective Study/collection_stud_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    if deviceManager.getDevice('key_resp_12') is None:
        # initialise key_resp_12
        key_resp_12 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_12',
        )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
        )
    if deviceManager.getDevice('key_resp_13') is None:
        # initialise key_resp_13
        key_resp_13 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_13',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_11') is None:
        # initialise key_resp_11
        key_resp_11 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_11',
        )
    if deviceManager.getDevice('key_resp_7') is None:
        # initialise key_resp_7
        key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_7',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_8') is None:
        # initialise key_resp_8
        key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_8',
        )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    if deviceManager.getDevice('key_resp_6') is None:
        # initialise key_resp_6
        key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6',
        )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instruction_BaseRate" ---
    # Run 'Begin Experiment' code from code_6
    ### LSL
    # ADD LSL STREAM  
    from pylsl import StreamInfo, StreamOutlet
    # Set up LabStreamingLayer stream.
    info = StreamInfo(name='LSLMarkersInletStreamName1', type='Markers', channel_count=1,
                      channel_format='int32', source_id='psychopy_stream_133938')
    outlet = StreamOutlet(info)  # Broadcast the stream.
    text = visual.TextStim(win=win, name='text',
        text="In the first experiment you will get to see information about the personality traits of a specific person and further information about a population group composition. \nYou will be asked to indicate to which population group the person most likely belongs to. \n\nYou will see each describtion part for 2sec and will have 7sec to give a response. \n\n\nYou can start the experiment pressing the 'space' button\n\n",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "FixationCross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "BaseRate_testtrial" ---
    text_38 = visual.TextStim(win=win, name='text_38',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BRT" ---
    text_40 = visual.TextStim(win=win, name='text_40',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "routine_1" ---
    text_41 = visual.TextStim(win=win, name='text_41',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BaseRate_testResponse" ---
    text_39 = visual.TextStim(win=win, name='text_39',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_12 = keyboard.Keyboard(deviceName='key_resp_12')
    # Run 'Begin Experiment' code from code_17
    if key_resp_12.status == STARTED: 
        outlet.push_sample(x=['BaseTest'])
    
    # --- Initialize components for Routine "Fixation" ---
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2', vertices='cross',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "BR1_2" ---
    text_44 = visual.TextStim(win=win, name='text_44',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_22
    outlet.push_sample(x=[1])  
    
    
    # --- Initialize components for Routine "BR2_2" ---
    text_45 = visual.TextStim(win=win, name='text_45',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_23
    outlet.push_sample(x=[1])  
    
    
    # --- Initialize components for Routine "BR3_2" ---
    text_46 = visual.TextStim(win=win, name='text_46',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_24
    outlet.push_sample(x=[1])  
    
    
    # --- Initialize components for Routine "BRQ" ---
    text_47 = visual.TextStim(win=win, name='text_47',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_25
    outlet.push_sample(x=[100])  
    
    
    # --- Initialize components for Routine "Instruction_Syl_2" ---
    text_21 = visual.TextStim(win=win, name='text_21',
        text="In this experiment you will be given four problems. In each case, you will be given a prose passage to read and asked if a certain conclusion may be logically deduced from it. You should answer this question on the assumption that all the information given in the passage is, in fact, true. If you judge that the conclusion necessarily follows from the statements in the passage, you should answer\n\n'yes,' otherwise 'no.'\nYou can start the experiment by pressing the 'space' bar",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    
    # --- Initialize components for Routine "ST1" ---
    text_48 = visual.TextStim(win=win, name='text_48',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "ST2" ---
    text_49 = visual.TextStim(win=win, name='text_49',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "ST3" ---
    text_50 = visual.TextStim(win=win, name='text_50',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "SylTestResponse" ---
    text_42 = visual.TextStim(win=win, name='text_42',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_13 = keyboard.Keyboard(deviceName='key_resp_13')
    
    # --- Initialize components for Routine "STA1" ---
    text_51 = visual.TextStim(win=win, name='text_51',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "STA2_2" ---
    text_53 = visual.TextStim(win=win, name='text_53',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "STA3" ---
    text_52 = visual.TextStim(win=win, name='text_52',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question_logicalconclu" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "FixationCross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "STB1" ---
    text_55 = visual.TextStim(win=win, name='text_55',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "STB2" ---
    text_56 = visual.TextStim(win=win, name='text_56',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "STB3" ---
    text_57 = visual.TextStim(win=win, name='text_57',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question_locical" ---
    text_35 = visual.TextStim(win=win, name='text_35',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_11 = keyboard.Keyboard(deviceName='key_resp_11')
    
    # --- Initialize components for Routine "Instruction_CRT" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text="Please type the number that you think solves the question. \nIf you want to skip to the next trial please press the 'space' bar.\n\nYou can start the experiment by pressing the 'space' bar\n",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_7 = keyboard.Keyboard(deviceName='key_resp_7')
    
    # --- Initialize components for Routine "break_lsl" ---
    text_31 = visual.TextStim(win=win, name='text_31',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial_CRT" ---
    text_11 = visual.TextStim(win=win, name='text_11',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, -0.3),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-2, autoLog=True,
    )
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "Fake_TrueHeadlines" ---
    text_26 = visual.TextStim(win=win, name='text_26',
        text="In the upcoming experiment, you will be shown a series of news headlines. Initially, you will need to assess and indicate whether you believe these headlines are true or false.\n\nYou can start the experiment by pressing on the 'space'",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_8 = keyboard.Keyboard(deviceName='key_resp_8')
    
    # --- Initialize components for Routine "shortbreak_2" ---
    text_34 = visual.TextStim(win=win, name='text_34',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial_fakenews" ---
    text_27 = visual.TextStim(win=win, name='text_27',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "shortbreak" ---
    text_33 = visual.TextStim(win=win, name='text_33',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "PressNext" ---
    text_32 = visual.TextStim(win=win, name='text_32',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    
    # --- Initialize components for Routine "Participant_feedback" ---
    text_28 = visual.TextStim(win=win, name='text_28',
        text='Do you belief this statement?',
        font='Open Sans',
        pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_6 = visual.Slider(win=win, name='slider_6',
        startValue=3, size=(1.0, 0.05), pos=(0, 0.3), units=win.units,
        labels=('Definitely Flase','Probably False','','Probably True','Definitley True'), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    text_29 = visual.TextStim(win=win, name='text_29',
        text='How confident are you in your answer?',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    slider_7 = visual.Slider(win=win, name='slider_7',
        startValue=None, size=(1.0, 0.05), pos=(0, 0), units=win.units,
        labels=(0, 10,20, 30, 40, 50,60,70,80,90,100), ticks=(1, 2, 3, 4, 5,6,7,8,9,10,11), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    text_30 = visual.TextStim(win=win, name='text_30',
        text='How knowledgable are you on the topic?',
        font='Open Sans',
        pos=(0, -0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    slider_8 = visual.Slider(win=win, name='slider_8',
        startValue=3, size=(1.0, 0.05), pos=(0, -0.3), units=win.units,
        labels=('Not at all knowledgable','Somewhat not knowledgable','','Somewhat knowledgable','Very much knowldgable'), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    button_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0.6, -0.3),
        letterHeight=0.02,
        size=(0.1, 0.05), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_3',
        depth=-6
    )
    button_3.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "Instruction_Policyst" ---
    text_22 = visual.TextStim(win=win, name='text_22',
        text="In the upcoming experiment, you will be shown a series of news headlines. Initially, you will need to assess and indicate whether you believe these headlines are true or false. Following your initial response, you will receive community notes related to each headline. After reviewing these notes, you will be asked to re-evaluate and provide your final judgment on the truthfulness of each statement.\n\nYou can start the experiment by pressing on the 'space'",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_6 = keyboard.Keyboard(deviceName='key_resp_6')
    
    # --- Initialize components for Routine "FixationCross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "P1" ---
    text_58 = visual.TextStim(win=win, name='text_58',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial5" ---
    text_13 = visual.TextStim(win=win, name='text_13',
        text='',
        font='Open Sans',
        pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_14 = visual.TextStim(win=win, name='text_14',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider = visual.Slider(win=win, name='slider',
        startValue=3, size=(1.0, 0.05), pos=(0, -0.1), units=win.units,
        labels=('Definitely Flase','Probably False','','Probably True','Definitley True'), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_15 = visual.TextStim(win=win, name='text_15',
        text='',
        font='Open Sans',
        pos=(0, -0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    slider_2 = visual.Slider(win=win, name='slider_2',
        startValue=0, size=(1.0, 0.05), pos=(0, -0.3), units=win.units,
        labels=(0, 10,20, 30, 40, 50,60,70,80,90,100), ticks=(1, 2, 3, 4, 5,6,7,8,9,10,11), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    
    # --- Initialize components for Routine "break_2" ---
    text_59 = visual.TextStim(win=win, name='text_59',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "P2" ---
    text_62 = visual.TextStim(win=win, name='text_62',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_60 = visual.TextStim(win=win, name='text_60',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Revision" ---
    text_37 = visual.TextStim(win=win, name='text_37',
        text='',
        font='Open Sans',
        pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_17 = visual.TextStim(win=win, name='text_17',
        text='',
        font='Open Sans',
        pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_3 = visual.Slider(win=win, name='slider_3',
        startValue=3, size=(1.0, 0.05), pos=(0,0.2), units=win.units,
        labels=('Definitely Flase','Probably False','','Probably True','Definitley True'), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_18 = visual.TextStim(win=win, name='text_18',
        text='',
        font='Open Sans',
        pos=(0, 0.0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    slider_4 = visual.Slider(win=win, name='slider_4',
        startValue=0, size=(1.0, 0.05), pos=(0, -0.1), units=win.units,
        labels=(0, 10,20, 30, 40, 50,60,70,80,90,100), ticks=(1, 2, 3, 4, 5,6,7,8,9,10,11), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    text_19 = visual.TextStim(win=win, name='text_19',
        text='',
        font='Open Sans',
        pos=(0, -0.3), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    slider_5 = visual.Slider(win=win, name='slider_5',
        startValue=5, size=(1.0, 0.05), pos=(0, -0.4), units=win.units,
        labels=('Not at all knowledgable','Somewhat not knowledgable','','Somewhat knowledgable','Very much knowldgable'), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    button = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0.6, -0.3),
        letterHeight=0.02,
        size=(0.1, 0.05), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-7
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "break_2" ---
    text_59 = visual.TextStim(win=win, name='text_59',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "P3" ---
    text_63 = visual.TextStim(win=win, name='text_63',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_61 = visual.TextStim(win=win, name='text_61',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Beginning_questionnaire" ---
    text_25 = visual.TextStim(win=win, name='text_25',
        text='Before we start the experiment, please indicate which of the following best describes your usual political stance?\n\na) Republican\nb) Democratic\nc) Independent\nd) Other\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
    # --- Initialize components for Routine "EndExperiment" ---
    text_20 = visual.TextStim(win=win, name='text_20',
        text='End of Experiment',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instruction_BaseRate" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_BaseRate.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # keep track of which components have finished
    Instruction_BaseRateComponents = [text, key_resp_4]
    for thisComponent in Instruction_BaseRateComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction_BaseRate" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction_BaseRateComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction_BaseRate" ---
    for thisComponent in Instruction_BaseRateComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruction_BaseRate.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    # Run 'End Routine' code from code_8
    outlet.push_sample(x=[100])  
    
    thisExp.nextEntry()
    # the Routine "Instruction_BaseRate" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    BaseRate_test = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BaseRateTest.xlsx'),
        seed=None, name='BaseRate_test')
    thisExp.addLoop(BaseRate_test)  # add the loop to the experiment
    thisBaseRate_test = BaseRate_test.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBaseRate_test.rgb)
    if thisBaseRate_test != None:
        for paramName in thisBaseRate_test:
            globals()[paramName] = thisBaseRate_test[paramName]
    
    for thisBaseRate_test in BaseRate_test:
        currentLoop = BaseRate_test
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBaseRate_test.rgb)
        if thisBaseRate_test != None:
            for paramName in thisBaseRate_test:
                globals()[paramName] = thisBaseRate_test[paramName]
        
        # --- Prepare to start Routine "FixationCross" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('FixationCross.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        FixationCrossComponents = [polygon]
        for thisComponent in FixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FixationCross" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FixationCross" ---
        for thisComponent in FixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('FixationCross.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "BaseRate_testtrial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BaseRate_testtrial.started', globalClock.getTime(format='float'))
        text_38.setText(BaseRateTest
        )
        # keep track of which components have finished
        BaseRate_testtrialComponents = [text_38]
        for thisComponent in BaseRate_testtrialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BaseRate_testtrial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_38* updates
            
            # if text_38 is starting this frame...
            if text_38.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_38.frameNStart = frameN  # exact frame index
                text_38.tStart = t  # local t and not account for scr refresh
                text_38.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_38, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_38.started')
                # update status
                text_38.status = STARTED
                text_38.setAutoDraw(True)
            
            # if text_38 is active this frame...
            if text_38.status == STARTED:
                # update params
                pass
            
            # if text_38 is stopping this frame...
            if text_38.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_38.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_38.tStop = t  # not accounting for scr refresh
                    text_38.tStopRefresh = tThisFlipGlobal  # on global time
                    text_38.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_38.stopped')
                    # update status
                    text_38.status = FINISHED
                    text_38.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BaseRate_testtrialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BaseRate_testtrial" ---
        for thisComponent in BaseRate_testtrialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BaseRate_testtrial.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "BRT" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BRT.started', globalClock.getTime(format='float'))
        text_40.setText(BaseRateDescr
        )
        # keep track of which components have finished
        BRTComponents = [text_40]
        for thisComponent in BRTComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BRT" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_40* updates
            
            # if text_40 is starting this frame...
            if text_40.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_40.frameNStart = frameN  # exact frame index
                text_40.tStart = t  # local t and not account for scr refresh
                text_40.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_40, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_40.started')
                # update status
                text_40.status = STARTED
                text_40.setAutoDraw(True)
            
            # if text_40 is active this frame...
            if text_40.status == STARTED:
                # update params
                pass
            
            # if text_40 is stopping this frame...
            if text_40.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_40.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_40.tStop = t  # not accounting for scr refresh
                    text_40.tStopRefresh = tThisFlipGlobal  # on global time
                    text_40.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_40.stopped')
                    # update status
                    text_40.status = FINISHED
                    text_40.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BRTComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BRT" ---
        for thisComponent in BRTComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BRT.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "routine_1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('routine_1.started', globalClock.getTime(format='float'))
        text_41.setText(BaseRateNr)
        # keep track of which components have finished
        routine_1Components = [text_41]
        for thisComponent in routine_1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_41* updates
            
            # if text_41 is starting this frame...
            if text_41.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_41.frameNStart = frameN  # exact frame index
                text_41.tStart = t  # local t and not account for scr refresh
                text_41.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_41, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_41.started')
                # update status
                text_41.status = STARTED
                text_41.setAutoDraw(True)
            
            # if text_41 is active this frame...
            if text_41.status == STARTED:
                # update params
                pass
            
            # if text_41 is stopping this frame...
            if text_41.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_41.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_41.tStop = t  # not accounting for scr refresh
                    text_41.tStopRefresh = tThisFlipGlobal  # on global time
                    text_41.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_41.stopped')
                    # update status
                    text_41.status = FINISHED
                    text_41.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in routine_1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_1" ---
        for thisComponent in routine_1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('routine_1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "BaseRate_testResponse" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BaseRate_testResponse.started', globalClock.getTime(format='float'))
        text_39.setText(BaseRateResponse)
        # create starting attributes for key_resp_12
        key_resp_12.keys = []
        key_resp_12.rt = []
        _key_resp_12_allKeys = []
        # keep track of which components have finished
        BaseRate_testResponseComponents = [text_39, key_resp_12]
        for thisComponent in BaseRate_testResponseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BaseRate_testResponse" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_39* updates
            
            # if text_39 is starting this frame...
            if text_39.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_39.frameNStart = frameN  # exact frame index
                text_39.tStart = t  # local t and not account for scr refresh
                text_39.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_39, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_39.started')
                # update status
                text_39.status = STARTED
                text_39.setAutoDraw(True)
            
            # if text_39 is active this frame...
            if text_39.status == STARTED:
                # update params
                pass
            
            # *key_resp_12* updates
            waitOnFlip = False
            
            # if key_resp_12 is starting this frame...
            if key_resp_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_12.frameNStart = frameN  # exact frame index
                key_resp_12.tStart = t  # local t and not account for scr refresh
                key_resp_12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_12.started')
                # update status
                key_resp_12.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_12.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_12.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_12.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_12_allKeys.extend(theseKeys)
                if len(_key_resp_12_allKeys):
                    key_resp_12.keys = _key_resp_12_allKeys[-1].name  # just the last key pressed
                    key_resp_12.rt = _key_resp_12_allKeys[-1].rt
                    key_resp_12.duration = _key_resp_12_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BaseRate_testResponseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BaseRate_testResponse" ---
        for thisComponent in BaseRate_testResponseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BaseRate_testResponse.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_12.keys in ['', [], None]:  # No response was made
            key_resp_12.keys = None
        BaseRate_test.addData('key_resp_12.keys',key_resp_12.keys)
        if key_resp_12.keys != None:  # we had a response
            BaseRate_test.addData('key_resp_12.rt', key_resp_12.rt)
            BaseRate_test.addData('key_resp_12.duration', key_resp_12.duration)
        # the Routine "BaseRate_testResponse" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'BaseRate_test'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_18 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BaseRateinfo.xlsx'),
        seed=None, name='trials_18')
    thisExp.addLoop(trials_18)  # add the loop to the experiment
    thisTrial_18 = trials_18.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_18.rgb)
    if thisTrial_18 != None:
        for paramName in thisTrial_18:
            globals()[paramName] = thisTrial_18[paramName]
    
    for thisTrial_18 in trials_18:
        currentLoop = trials_18
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_18.rgb)
        if thisTrial_18 != None:
            for paramName in thisTrial_18:
                globals()[paramName] = thisTrial_18[paramName]
        
        # --- Prepare to start Routine "Fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fixation.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        FixationComponents = [polygon_2]
        for thisComponent in FixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # if polygon_2 is stopping this frame...
            if polygon_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_2.tStop = t  # not accounting for scr refresh
                    polygon_2.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                    # update status
                    polygon_2.status = FINISHED
                    polygon_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation" ---
        for thisComponent in FixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fixation.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "BR1_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BR1_2.started', globalClock.getTime(format='float'))
        text_44.setText(BR1
        )
        # keep track of which components have finished
        BR1_2Components = [text_44]
        for thisComponent in BR1_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BR1_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_44* updates
            
            # if text_44 is starting this frame...
            if text_44.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_44.frameNStart = frameN  # exact frame index
                text_44.tStart = t  # local t and not account for scr refresh
                text_44.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_44, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_44.started')
                # update status
                text_44.status = STARTED
                text_44.setAutoDraw(True)
            
            # if text_44 is active this frame...
            if text_44.status == STARTED:
                # update params
                pass
            
            # if text_44 is stopping this frame...
            if text_44.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_44.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_44.tStop = t  # not accounting for scr refresh
                    text_44.tStopRefresh = tThisFlipGlobal  # on global time
                    text_44.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_44.stopped')
                    # update status
                    text_44.status = FINISHED
                    text_44.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BR1_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BR1_2" ---
        for thisComponent in BR1_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BR1_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "BR2_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BR2_2.started', globalClock.getTime(format='float'))
        text_45.setText(BR1+'\n'+ str(BR2))
        # keep track of which components have finished
        BR2_2Components = [text_45]
        for thisComponent in BR2_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BR2_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_45* updates
            
            # if text_45 is starting this frame...
            if text_45.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_45.frameNStart = frameN  # exact frame index
                text_45.tStart = t  # local t and not account for scr refresh
                text_45.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_45, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_45.started')
                # update status
                text_45.status = STARTED
                text_45.setAutoDraw(True)
            
            # if text_45 is active this frame...
            if text_45.status == STARTED:
                # update params
                pass
            
            # if text_45 is stopping this frame...
            if text_45.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_45.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_45.tStop = t  # not accounting for scr refresh
                    text_45.tStopRefresh = tThisFlipGlobal  # on global time
                    text_45.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_45.stopped')
                    # update status
                    text_45.status = FINISHED
                    text_45.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BR2_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BR2_2" ---
        for thisComponent in BR2_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BR2_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "BR3_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BR3_2.started', globalClock.getTime(format='float'))
        text_46.setText(BR1+'\n'+ str(BR2) + '\n'+ str(BR3))
        # keep track of which components have finished
        BR3_2Components = [text_46]
        for thisComponent in BR3_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BR3_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_46* updates
            
            # if text_46 is starting this frame...
            if text_46.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_46.frameNStart = frameN  # exact frame index
                text_46.tStart = t  # local t and not account for scr refresh
                text_46.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_46, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_46.started')
                # update status
                text_46.status = STARTED
                text_46.setAutoDraw(True)
            
            # if text_46 is active this frame...
            if text_46.status == STARTED:
                # update params
                pass
            
            # if text_46 is stopping this frame...
            if text_46.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_46.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_46.tStop = t  # not accounting for scr refresh
                    text_46.tStopRefresh = tThisFlipGlobal  # on global time
                    text_46.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_46.stopped')
                    # update status
                    text_46.status = FINISHED
                    text_46.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BR3_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BR3_2" ---
        for thisComponent in BR3_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BR3_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "BRQ" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BRQ.started', globalClock.getTime(format='float'))
        text_47.setText(baserate_response)
        # keep track of which components have finished
        BRQComponents = [text_47]
        for thisComponent in BRQComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BRQ" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_47* updates
            
            # if text_47 is starting this frame...
            if text_47.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_47.frameNStart = frameN  # exact frame index
                text_47.tStart = t  # local t and not account for scr refresh
                text_47.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_47, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_47.started')
                # update status
                text_47.status = STARTED
                text_47.setAutoDraw(True)
            
            # if text_47 is active this frame...
            if text_47.status == STARTED:
                # update params
                pass
            
            # if text_47 is stopping this frame...
            if text_47.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_47.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_47.tStop = t  # not accounting for scr refresh
                    text_47.tStopRefresh = tThisFlipGlobal  # on global time
                    text_47.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_47.stopped')
                    # update status
                    text_47.status = FINISHED
                    text_47.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BRQComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BRQ" ---
        for thisComponent in BRQComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BRQ.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_18'
    
    
    # --- Prepare to start Routine "Instruction_Syl_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_Syl_2.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from LSL_Start
    
    
    # create starting attributes for key_resp_5
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # keep track of which components have finished
    Instruction_Syl_2Components = [text_21, key_resp_5]
    for thisComponent in Instruction_Syl_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction_Syl_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_21* updates
        
        # if text_21 is starting this frame...
        if text_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_21.frameNStart = frameN  # exact frame index
            text_21.tStart = t  # local t and not account for scr refresh
            text_21.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_21, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_21.started')
            # update status
            text_21.status = STARTED
            text_21.setAutoDraw(True)
        
        # if text_21 is active this frame...
        if text_21.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_5.started')
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction_Syl_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction_Syl_2" ---
    for thisComponent in Instruction_Syl_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruction_Syl_2.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from LSL_Start
    outlet.push_sample(x=[100])  # Push event marker. Start experiment
    
    # check responses
    if key_resp_5.keys in ['', [], None]:  # No response was made
        key_resp_5.keys = None
    thisExp.addData('key_resp_5.keys',key_resp_5.keys)
    if key_resp_5.keys != None:  # we had a response
        thisExp.addData('key_resp_5.rt', key_resp_5.rt)
        thisExp.addData('key_resp_5.duration', key_resp_5.duration)
    thisExp.nextEntry()
    # the Routine "Instruction_Syl_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('Syltest.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "ST1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ST1.started', globalClock.getTime(format='float'))
        text_48.setText(ST1)
        # Run 'Begin Routine' code from code_26
        outlet.push_sample(x=[2])
        # keep track of which components have finished
        ST1Components = [text_48]
        for thisComponent in ST1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ST1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_48* updates
            
            # if text_48 is starting this frame...
            if text_48.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_48.frameNStart = frameN  # exact frame index
                text_48.tStart = t  # local t and not account for scr refresh
                text_48.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_48, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_48.started')
                # update status
                text_48.status = STARTED
                text_48.setAutoDraw(True)
            
            # if text_48 is active this frame...
            if text_48.status == STARTED:
                # update params
                pass
            
            # if text_48 is stopping this frame...
            if text_48.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_48.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_48.tStop = t  # not accounting for scr refresh
                    text_48.tStopRefresh = tThisFlipGlobal  # on global time
                    text_48.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_48.stopped')
                    # update status
                    text_48.status = FINISHED
                    text_48.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ST1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ST1" ---
        for thisComponent in ST1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ST1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "ST2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ST2.started', globalClock.getTime(format='float'))
        text_49.setText(ST1+'\n'+ str(ST2))
        # Run 'Begin Routine' code from code_27
        outlet.push_sample(x=[2])
        # keep track of which components have finished
        ST2Components = [text_49]
        for thisComponent in ST2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ST2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_49* updates
            
            # if text_49 is starting this frame...
            if text_49.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_49.frameNStart = frameN  # exact frame index
                text_49.tStart = t  # local t and not account for scr refresh
                text_49.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_49, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_49.started')
                # update status
                text_49.status = STARTED
                text_49.setAutoDraw(True)
            
            # if text_49 is active this frame...
            if text_49.status == STARTED:
                # update params
                pass
            
            # if text_49 is stopping this frame...
            if text_49.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_49.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_49.tStop = t  # not accounting for scr refresh
                    text_49.tStopRefresh = tThisFlipGlobal  # on global time
                    text_49.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_49.stopped')
                    # update status
                    text_49.status = FINISHED
                    text_49.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ST2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ST2" ---
        for thisComponent in ST2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ST2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "ST3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ST3.started', globalClock.getTime(format='float'))
        text_50.setText(ST1+'\n'+ str(ST2)+'\n'+ str(ST3) )
        # Run 'Begin Routine' code from code_28
        outlet.push_sample(x=[2])
        # keep track of which components have finished
        ST3Components = [text_50]
        for thisComponent in ST3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ST3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_50* updates
            
            # if text_50 is starting this frame...
            if text_50.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_50.frameNStart = frameN  # exact frame index
                text_50.tStart = t  # local t and not account for scr refresh
                text_50.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_50, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_50.started')
                # update status
                text_50.status = STARTED
                text_50.setAutoDraw(True)
            
            # if text_50 is active this frame...
            if text_50.status == STARTED:
                # update params
                pass
            
            # if text_50 is stopping this frame...
            if text_50.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_50.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_50.tStop = t  # not accounting for scr refresh
                    text_50.tStopRefresh = tThisFlipGlobal  # on global time
                    text_50.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_50.stopped')
                    # update status
                    text_50.status = FINISHED
                    text_50.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ST3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ST3" ---
        for thisComponent in ST3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ST3.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "SylTestResponse" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SylTestResponse.started', globalClock.getTime(format='float'))
        text_42.setText('Does the conclusion follow logically?\n\na) yes\nb) no')
        # create starting attributes for key_resp_13
        key_resp_13.keys = []
        key_resp_13.rt = []
        _key_resp_13_allKeys = []
        # Run 'Begin Routine' code from code_18
        outlet.push_sample(x=[200])
        # keep track of which components have finished
        SylTestResponseComponents = [text_42, key_resp_13]
        for thisComponent in SylTestResponseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SylTestResponse" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_42* updates
            
            # if text_42 is starting this frame...
            if text_42.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_42.frameNStart = frameN  # exact frame index
                text_42.tStart = t  # local t and not account for scr refresh
                text_42.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_42, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_42.started')
                # update status
                text_42.status = STARTED
                text_42.setAutoDraw(True)
            
            # if text_42 is active this frame...
            if text_42.status == STARTED:
                # update params
                pass
            
            # *key_resp_13* updates
            waitOnFlip = False
            
            # if key_resp_13 is starting this frame...
            if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_13.frameNStart = frameN  # exact frame index
                key_resp_13.tStart = t  # local t and not account for scr refresh
                key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_13.started')
                # update status
                key_resp_13.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_13.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_13.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_13_allKeys.extend(theseKeys)
                if len(_key_resp_13_allKeys):
                    key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                    key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                    key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SylTestResponseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SylTestResponse" ---
        for thisComponent in SylTestResponseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SylTestResponse.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_13.keys in ['', [], None]:  # No response was made
            key_resp_13.keys = None
        trials.addData('key_resp_13.keys',key_resp_13.keys)
        if key_resp_13.keys != None:  # we had a response
            trials.addData('key_resp_13.rt', key_resp_13.rt)
            trials.addData('key_resp_13.duration', key_resp_13.duration)
        # the Routine "SylTestResponse" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 5.0 repeats of 'trials'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('SyllologisticTask.xlsx'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "STA1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('STA1.started', globalClock.getTime(format='float'))
        text_51.setText(STA1)
        # Run 'Begin Routine' code from code_29
        outlet.push_sample(x=[3])  
        
        # keep track of which components have finished
        STA1Components = [text_51]
        for thisComponent in STA1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "STA1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_51* updates
            
            # if text_51 is starting this frame...
            if text_51.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_51.frameNStart = frameN  # exact frame index
                text_51.tStart = t  # local t and not account for scr refresh
                text_51.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_51, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_51.started')
                # update status
                text_51.status = STARTED
                text_51.setAutoDraw(True)
            
            # if text_51 is active this frame...
            if text_51.status == STARTED:
                # update params
                pass
            
            # if text_51 is stopping this frame...
            if text_51.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_51.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_51.tStop = t  # not accounting for scr refresh
                    text_51.tStopRefresh = tThisFlipGlobal  # on global time
                    text_51.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_51.stopped')
                    # update status
                    text_51.status = FINISHED
                    text_51.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in STA1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "STA1" ---
        for thisComponent in STA1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('STA1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "STA2_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('STA2_2.started', globalClock.getTime(format='float'))
        text_53.setText(STA1+'\n'+ str(STA2))
        # Run 'Begin Routine' code from code_30
        outlet.push_sample(x=[3])  
        
        # keep track of which components have finished
        STA2_2Components = [text_53]
        for thisComponent in STA2_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "STA2_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_53* updates
            
            # if text_53 is starting this frame...
            if text_53.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_53.frameNStart = frameN  # exact frame index
                text_53.tStart = t  # local t and not account for scr refresh
                text_53.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_53, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_53.started')
                # update status
                text_53.status = STARTED
                text_53.setAutoDraw(True)
            
            # if text_53 is active this frame...
            if text_53.status == STARTED:
                # update params
                pass
            
            # if text_53 is stopping this frame...
            if text_53.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_53.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_53.tStop = t  # not accounting for scr refresh
                    text_53.tStopRefresh = tThisFlipGlobal  # on global time
                    text_53.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_53.stopped')
                    # update status
                    text_53.status = FINISHED
                    text_53.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in STA2_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "STA2_2" ---
        for thisComponent in STA2_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('STA2_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "STA3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('STA3.started', globalClock.getTime(format='float'))
        text_52.setText(STA1+'\n'+ str(STA2)+'\n'+ str(STA3) )
        # Run 'Begin Routine' code from code_31
        outlet.push_sample(x=[3])  
        
        # keep track of which components have finished
        STA3Components = [text_52]
        for thisComponent in STA3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "STA3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_52* updates
            
            # if text_52 is starting this frame...
            if text_52.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_52.frameNStart = frameN  # exact frame index
                text_52.tStart = t  # local t and not account for scr refresh
                text_52.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_52, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_52.started')
                # update status
                text_52.status = STARTED
                text_52.setAutoDraw(True)
            
            # if text_52 is active this frame...
            if text_52.status == STARTED:
                # update params
                pass
            
            # if text_52 is stopping this frame...
            if text_52.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_52.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_52.tStop = t  # not accounting for scr refresh
                    text_52.tStopRefresh = tThisFlipGlobal  # on global time
                    text_52.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_52.stopped')
                    # update status
                    text_52.status = FINISHED
                    text_52.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in STA3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "STA3" ---
        for thisComponent in STA3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('STA3.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "question_logicalconclu" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('question_logicalconclu.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_7
        outlet.push_sample(x=[30])
        text_6.setText('Does the conclusion follow logically?\n\na) yes\nb) no')
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        question_logicalconcluComponents = [text_6, key_resp_2]
        for thisComponent in question_logicalconcluComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_logicalconclu" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_2.keys == str(key_ab1)) or (key_resp_2.keys == key_ab1):
                        key_resp_2.corr = 1
                    else:
                        key_resp_2.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in question_logicalconcluComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_logicalconclu" ---
        for thisComponent in question_logicalconcluComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('question_logicalconclu.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
            # was no response the correct answer?!
            if str(key_ab1).lower() == 'none':
               key_resp_2.corr = 1;  # correct non-response
            else:
               key_resp_2.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('key_resp_2.keys',key_resp_2.keys)
        trials_2.addData('key_resp_2.corr', key_resp_2.corr)
        if key_resp_2.keys != None:  # we had a response
            trials_2.addData('key_resp_2.rt', key_resp_2.rt)
            trials_2.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "question_logicalconclu" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_2'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('TypeB.xlsx'),
        seed=None, name='trials_3')
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "FixationCross" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('FixationCross.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        FixationCrossComponents = [polygon]
        for thisComponent in FixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FixationCross" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FixationCross" ---
        for thisComponent in FixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('FixationCross.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "STB1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('STB1.started', globalClock.getTime(format='float'))
        text_55.setText(STB1)
        # Run 'Begin Routine' code from code_32
        outlet.push_sample(x=[4])  
        
        # keep track of which components have finished
        STB1Components = [text_55]
        for thisComponent in STB1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "STB1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_55* updates
            
            # if text_55 is starting this frame...
            if text_55.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_55.frameNStart = frameN  # exact frame index
                text_55.tStart = t  # local t and not account for scr refresh
                text_55.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_55, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_55.started')
                # update status
                text_55.status = STARTED
                text_55.setAutoDraw(True)
            
            # if text_55 is active this frame...
            if text_55.status == STARTED:
                # update params
                pass
            
            # if text_55 is stopping this frame...
            if text_55.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_55.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_55.tStop = t  # not accounting for scr refresh
                    text_55.tStopRefresh = tThisFlipGlobal  # on global time
                    text_55.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_55.stopped')
                    # update status
                    text_55.status = FINISHED
                    text_55.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in STB1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "STB1" ---
        for thisComponent in STB1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('STB1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "STB2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('STB2.started', globalClock.getTime(format='float'))
        text_56.setText(STB1+'\n'+ str(STB2))
        # Run 'Begin Routine' code from code_33
        outlet.push_sample(x=[4])  
        
        # keep track of which components have finished
        STB2Components = [text_56]
        for thisComponent in STB2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "STB2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_56* updates
            
            # if text_56 is starting this frame...
            if text_56.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_56.frameNStart = frameN  # exact frame index
                text_56.tStart = t  # local t and not account for scr refresh
                text_56.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_56, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_56.started')
                # update status
                text_56.status = STARTED
                text_56.setAutoDraw(True)
            
            # if text_56 is active this frame...
            if text_56.status == STARTED:
                # update params
                pass
            
            # if text_56 is stopping this frame...
            if text_56.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_56.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_56.tStop = t  # not accounting for scr refresh
                    text_56.tStopRefresh = tThisFlipGlobal  # on global time
                    text_56.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_56.stopped')
                    # update status
                    text_56.status = FINISHED
                    text_56.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in STB2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "STB2" ---
        for thisComponent in STB2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('STB2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "STB3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('STB3.started', globalClock.getTime(format='float'))
        text_57.setText(STB1+'\n'+ str(STB2)+'\n'+ str(STB3) )
        # Run 'Begin Routine' code from code_34
        outlet.push_sample(x=[4])  
        
        # keep track of which components have finished
        STB3Components = [text_57]
        for thisComponent in STB3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "STB3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_57* updates
            
            # if text_57 is starting this frame...
            if text_57.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_57.frameNStart = frameN  # exact frame index
                text_57.tStart = t  # local t and not account for scr refresh
                text_57.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_57, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_57.started')
                # update status
                text_57.status = STARTED
                text_57.setAutoDraw(True)
            
            # if text_57 is active this frame...
            if text_57.status == STARTED:
                # update params
                pass
            
            # if text_57 is stopping this frame...
            if text_57.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_57.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_57.tStop = t  # not accounting for scr refresh
                    text_57.tStopRefresh = tThisFlipGlobal  # on global time
                    text_57.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_57.stopped')
                    # update status
                    text_57.status = FINISHED
                    text_57.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in STB3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "STB3" ---
        for thisComponent in STB3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('STB3.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "question_locical" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('question_locical.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_16
        outlet.push_sample(x=[40])  
        
        text_35.setText('Does the conclusion follow logically?\n\na) yes\nb) no')
        # create starting attributes for key_resp_11
        key_resp_11.keys = []
        key_resp_11.rt = []
        _key_resp_11_allKeys = []
        # keep track of which components have finished
        question_locicalComponents = [text_35, key_resp_11]
        for thisComponent in question_locicalComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_locical" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_35* updates
            
            # if text_35 is starting this frame...
            if text_35.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_35.frameNStart = frameN  # exact frame index
                text_35.tStart = t  # local t and not account for scr refresh
                text_35.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_35, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_35.started')
                # update status
                text_35.status = STARTED
                text_35.setAutoDraw(True)
            
            # if text_35 is active this frame...
            if text_35.status == STARTED:
                # update params
                pass
            
            # *key_resp_11* updates
            waitOnFlip = False
            
            # if key_resp_11 is starting this frame...
            if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_11.frameNStart = frameN  # exact frame index
                key_resp_11.tStart = t  # local t and not account for scr refresh
                key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_11.started')
                # update status
                key_resp_11.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_11.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_11.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_11_allKeys.extend(theseKeys)
                if len(_key_resp_11_allKeys):
                    key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                    key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                    key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_11.keys == str('key_ab2')) or (key_resp_11.keys == 'key_ab2'):
                        key_resp_11.corr = 1
                    else:
                        key_resp_11.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in question_locicalComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_locical" ---
        for thisComponent in question_locicalComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('question_locical.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_11.keys in ['', [], None]:  # No response was made
            key_resp_11.keys = None
            # was no response the correct answer?!
            if str('key_ab2').lower() == 'none':
               key_resp_11.corr = 1;  # correct non-response
            else:
               key_resp_11.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_3 (TrialHandler)
        trials_3.addData('key_resp_11.keys',key_resp_11.keys)
        trials_3.addData('key_resp_11.corr', key_resp_11.corr)
        if key_resp_11.keys != None:  # we had a response
            trials_3.addData('key_resp_11.rt', key_resp_11.rt)
            trials_3.addData('key_resp_11.duration', key_resp_11.duration)
        # the Routine "question_locical" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_3'
    
    
    # --- Prepare to start Routine "Instruction_CRT" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_CRT.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from code_9
    outlet.push_sample(x=[100])
    # create starting attributes for key_resp_7
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    # keep track of which components have finished
    Instruction_CRTComponents = [text_9, key_resp_7]
    for thisComponent in Instruction_CRTComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction_CRT" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_9.started')
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        
        # *key_resp_7* updates
        waitOnFlip = False
        
        # if key_resp_7 is starting this frame...
        if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_7.frameNStart = frameN  # exact frame index
            key_resp_7.tStart = t  # local t and not account for scr refresh
            key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_7.started')
            # update status
            key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_7_allKeys.extend(theseKeys)
            if len(_key_resp_7_allKeys):
                key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction_CRTComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction_CRT" ---
    for thisComponent in Instruction_CRTComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruction_CRT.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "Instruction_CRT" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_8 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('CRT_task.xlsx'),
        seed=None, name='trials_8')
    thisExp.addLoop(trials_8)  # add the loop to the experiment
    thisTrial_8 = trials_8.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_8.rgb)
    if thisTrial_8 != None:
        for paramName in thisTrial_8:
            globals()[paramName] = thisTrial_8[paramName]
    
    for thisTrial_8 in trials_8:
        currentLoop = trials_8
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_8.rgb)
        if thisTrial_8 != None:
            for paramName in thisTrial_8:
                globals()[paramName] = thisTrial_8[paramName]
        
        # --- Prepare to start Routine "break_lsl" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_lsl.started', globalClock.getTime(format='float'))
        text_31.setText('')
        # keep track of which components have finished
        break_lslComponents = [text_31]
        for thisComponent in break_lslComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_lsl" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_31* updates
            
            # if text_31 is starting this frame...
            if text_31.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_31.frameNStart = frameN  # exact frame index
                text_31.tStart = t  # local t and not account for scr refresh
                text_31.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_31, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_31.started')
                # update status
                text_31.status = STARTED
                text_31.setAutoDraw(True)
            
            # if text_31 is active this frame...
            if text_31.status == STARTED:
                # update params
                pass
            
            # if text_31 is stopping this frame...
            if text_31.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_31.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_31.tStop = t  # not accounting for scr refresh
                    text_31.tStopRefresh = tThisFlipGlobal  # on global time
                    text_31.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_31.stopped')
                    # update status
                    text_31.status = FINISHED
                    text_31.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_lslComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_lsl" ---
        for thisComponent in break_lslComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_lsl.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_CRT" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_CRT.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from LSL_CRT
        outlet.push_sample(x=[5])
        text_11.setText(CRT)
        textbox.reset()
        textbox.setText('')
        textbox.setPlaceholder('')
        # create starting attributes for key_resp_3
        key_resp_3.keys = []
        key_resp_3.rt = []
        _key_resp_3_allKeys = []
        # keep track of which components have finished
        trial_CRTComponents = [text_11, textbox, key_resp_3]
        for thisComponent in trial_CRTComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_CRT" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_11* updates
            
            # if text_11 is starting this frame...
            if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_11.frameNStart = frameN  # exact frame index
                text_11.tStart = t  # local t and not account for scr refresh
                text_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_11.started')
                # update status
                text_11.status = STARTED
                text_11.setAutoDraw(True)
            
            # if text_11 is active this frame...
            if text_11.status == STARTED:
                # update params
                pass
            
            # *textbox* updates
            
            # if textbox is starting this frame...
            if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox.frameNStart = frameN  # exact frame index
                textbox.tStart = t  # local t and not account for scr refresh
                textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox.started')
                # update status
                textbox.status = STARTED
                textbox.setAutoDraw(True)
            
            # if textbox is active this frame...
            if textbox.status == STARTED:
                # update params
                pass
            
            # *key_resp_3* updates
            waitOnFlip = False
            
            # if key_resp_3 is starting this frame...
            if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_3.frameNStart = frameN  # exact frame index
                key_resp_3.tStart = t  # local t and not account for scr refresh
                key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.started')
                # update status
                key_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_3_allKeys.extend(theseKeys)
                if len(_key_resp_3_allKeys):
                    key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                    key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                    key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_CRTComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_CRT" ---
        for thisComponent in trial_CRTComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_CRT.stopped', globalClock.getTime(format='float'))
        trials_8.addData('textbox.text',textbox.text)
        # check responses
        if key_resp_3.keys in ['', [], None]:  # No response was made
            key_resp_3.keys = None
        trials_8.addData('key_resp_3.keys',key_resp_3.keys)
        if key_resp_3.keys != None:  # we had a response
            trials_8.addData('key_resp_3.rt', key_resp_3.rt)
            trials_8.addData('key_resp_3.duration', key_resp_3.duration)
        # the Routine "trial_CRT" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_8'
    
    
    # --- Prepare to start Routine "Fake_TrueHeadlines" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Fake_TrueHeadlines.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from code_12
    outlet.push_sample(x=[100])
    # create starting attributes for key_resp_8
    key_resp_8.keys = []
    key_resp_8.rt = []
    _key_resp_8_allKeys = []
    # keep track of which components have finished
    Fake_TrueHeadlinesComponents = [text_26, key_resp_8]
    for thisComponent in Fake_TrueHeadlinesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Fake_TrueHeadlines" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_26* updates
        
        # if text_26 is starting this frame...
        if text_26.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_26.frameNStart = frameN  # exact frame index
            text_26.tStart = t  # local t and not account for scr refresh
            text_26.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_26, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_26.started')
            # update status
            text_26.status = STARTED
            text_26.setAutoDraw(True)
        
        # if text_26 is active this frame...
        if text_26.status == STARTED:
            # update params
            pass
        
        # *key_resp_8* updates
        waitOnFlip = False
        
        # if key_resp_8 is starting this frame...
        if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_8.frameNStart = frameN  # exact frame index
            key_resp_8.tStart = t  # local t and not account for scr refresh
            key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_8.started')
            # update status
            key_resp_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_8.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_8.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_8_allKeys.extend(theseKeys)
            if len(_key_resp_8_allKeys):
                key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Fake_TrueHeadlinesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fake_TrueHeadlines" ---
    for thisComponent in Fake_TrueHeadlinesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Fake_TrueHeadlines.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_8.keys in ['', [], None]:  # No response was made
        key_resp_8.keys = None
    thisExp.addData('key_resp_8.keys',key_resp_8.keys)
    if key_resp_8.keys != None:  # we had a response
        thisExp.addData('key_resp_8.rt', key_resp_8.rt)
        thisExp.addData('key_resp_8.duration', key_resp_8.duration)
    thisExp.nextEntry()
    # the Routine "Fake_TrueHeadlines" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_14 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('headlines_policystudy_tf.xlsx'),
        seed=None, name='trials_14')
    thisExp.addLoop(trials_14)  # add the loop to the experiment
    thisTrial_14 = trials_14.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_14.rgb)
    if thisTrial_14 != None:
        for paramName in thisTrial_14:
            globals()[paramName] = thisTrial_14[paramName]
    
    for thisTrial_14 in trials_14:
        currentLoop = trials_14
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_14.rgb)
        if thisTrial_14 != None:
            for paramName in thisTrial_14:
                globals()[paramName] = thisTrial_14[paramName]
        
        # --- Prepare to start Routine "shortbreak_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('shortbreak_2.started', globalClock.getTime(format='float'))
        text_34.setText('')
        # keep track of which components have finished
        shortbreak_2Components = [text_34]
        for thisComponent in shortbreak_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "shortbreak_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_34* updates
            
            # if text_34 is starting this frame...
            if text_34.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_34.frameNStart = frameN  # exact frame index
                text_34.tStart = t  # local t and not account for scr refresh
                text_34.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_34, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_34.started')
                # update status
                text_34.status = STARTED
                text_34.setAutoDraw(True)
            
            # if text_34 is active this frame...
            if text_34.status == STARTED:
                # update params
                pass
            
            # if text_34 is stopping this frame...
            if text_34.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_34.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_34.tStop = t  # not accounting for scr refresh
                    text_34.tStopRefresh = tThisFlipGlobal  # on global time
                    text_34.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_34.stopped')
                    # update status
                    text_34.status = FINISHED
                    text_34.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in shortbreak_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "shortbreak_2" ---
        for thisComponent in shortbreak_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('shortbreak_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_fakenews" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_fakenews.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_13
        words = headlines.split()
        for word in words:
            # Set text for this trial
            text.text = word
            outlet.push_sample(x=[6])  # Push event marker. Baseline==1
        
        
            # Show text for 300 ms = 18 frames on 60Hz monitor
            for frame in range(40):
                text.draw()
                win.flip()
        
        
        
        
        text_27.setText(headlines
        )
        # keep track of which components have finished
        trial_fakenewsComponents = [text_27]
        for thisComponent in trial_fakenewsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_fakenews" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_27* updates
            
            # if text_27 is starting this frame...
            if text_27.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_27.frameNStart = frameN  # exact frame index
                text_27.tStart = t  # local t and not account for scr refresh
                text_27.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_27, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_27.started')
                # update status
                text_27.status = STARTED
                text_27.setAutoDraw(True)
            
            # if text_27 is active this frame...
            if text_27.status == STARTED:
                # update params
                pass
            
            # if text_27 is stopping this frame...
            if text_27.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_27.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_27.tStop = t  # not accounting for scr refresh
                    text_27.tStopRefresh = tThisFlipGlobal  # on global time
                    text_27.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_27.stopped')
                    # update status
                    text_27.status = FINISHED
                    text_27.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_fakenewsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_fakenews" ---
        for thisComponent in trial_fakenewsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_fakenews.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "shortbreak" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('shortbreak.started', globalClock.getTime(format='float'))
        text_33.setText('')
        # keep track of which components have finished
        shortbreakComponents = [text_33]
        for thisComponent in shortbreakComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "shortbreak" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_33* updates
            
            # if text_33 is starting this frame...
            if text_33.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_33.frameNStart = frameN  # exact frame index
                text_33.tStart = t  # local t and not account for scr refresh
                text_33.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_33, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_33.started')
                # update status
                text_33.status = STARTED
                text_33.setAutoDraw(True)
            
            # if text_33 is active this frame...
            if text_33.status == STARTED:
                # update params
                pass
            
            # if text_33 is stopping this frame...
            if text_33.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_33.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_33.tStop = t  # not accounting for scr refresh
                    text_33.tStopRefresh = tThisFlipGlobal  # on global time
                    text_33.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_33.stopped')
                    # update status
                    text_33.status = FINISHED
                    text_33.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in shortbreakComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "shortbreak" ---
        for thisComponent in shortbreakComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('shortbreak.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "PressNext" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('PressNext.started', globalClock.getTime(format='float'))
        text_32.setText("Please Press the 'Space' button to continue")
        # create starting attributes for key_resp_9
        key_resp_9.keys = []
        key_resp_9.rt = []
        _key_resp_9_allKeys = []
        # keep track of which components have finished
        PressNextComponents = [text_32, key_resp_9]
        for thisComponent in PressNextComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "PressNext" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_32* updates
            
            # if text_32 is starting this frame...
            if text_32.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_32.frameNStart = frameN  # exact frame index
                text_32.tStart = t  # local t and not account for scr refresh
                text_32.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_32, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_32.started')
                # update status
                text_32.status = STARTED
                text_32.setAutoDraw(True)
            
            # if text_32 is active this frame...
            if text_32.status == STARTED:
                # update params
                pass
            
            # *key_resp_9* updates
            waitOnFlip = False
            
            # if key_resp_9 is starting this frame...
            if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_9.frameNStart = frameN  # exact frame index
                key_resp_9.tStart = t  # local t and not account for scr refresh
                key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_9.started')
                # update status
                key_resp_9.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_9.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_9.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_9_allKeys.extend(theseKeys)
                if len(_key_resp_9_allKeys):
                    key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                    key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                    key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in PressNextComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "PressNext" ---
        for thisComponent in PressNextComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('PressNext.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp_9.keys in ['', [], None]:  # No response was made
            key_resp_9.keys = None
        trials_14.addData('key_resp_9.keys',key_resp_9.keys)
        if key_resp_9.keys != None:  # we had a response
            trials_14.addData('key_resp_9.rt', key_resp_9.rt)
            trials_14.addData('key_resp_9.duration', key_resp_9.duration)
        # Run 'End Routine' code from code_11
        outlet.push_sample(x=[100])
        # the Routine "PressNext" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Participant_feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Participant_feedback.started', globalClock.getTime(format='float'))
        slider_6.reset()
        slider_7.reset()
        slider_8.reset()
        button_3.setText('Next')
        # reset button_3 to account for continued clicks & clear times on/off
        button_3.reset()
        # keep track of which components have finished
        Participant_feedbackComponents = [text_28, slider_6, text_29, slider_7, text_30, slider_8, button_3]
        for thisComponent in Participant_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Participant_feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_28* updates
            
            # if text_28 is starting this frame...
            if text_28.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_28.frameNStart = frameN  # exact frame index
                text_28.tStart = t  # local t and not account for scr refresh
                text_28.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_28, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_28.started')
                # update status
                text_28.status = STARTED
                text_28.setAutoDraw(True)
            
            # if text_28 is active this frame...
            if text_28.status == STARTED:
                # update params
                pass
            
            # *slider_6* updates
            
            # if slider_6 is starting this frame...
            if slider_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_6.frameNStart = frameN  # exact frame index
                slider_6.tStart = t  # local t and not account for scr refresh
                slider_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_6.started')
                # update status
                slider_6.status = STARTED
                slider_6.setAutoDraw(True)
            
            # if slider_6 is active this frame...
            if slider_6.status == STARTED:
                # update params
                pass
            
            # *text_29* updates
            
            # if text_29 is starting this frame...
            if text_29.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_29.frameNStart = frameN  # exact frame index
                text_29.tStart = t  # local t and not account for scr refresh
                text_29.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_29, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_29.started')
                # update status
                text_29.status = STARTED
                text_29.setAutoDraw(True)
            
            # if text_29 is active this frame...
            if text_29.status == STARTED:
                # update params
                pass
            
            # *slider_7* updates
            
            # if slider_7 is starting this frame...
            if slider_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_7.frameNStart = frameN  # exact frame index
                slider_7.tStart = t  # local t and not account for scr refresh
                slider_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_7.started')
                # update status
                slider_7.status = STARTED
                slider_7.setAutoDraw(True)
            
            # if slider_7 is active this frame...
            if slider_7.status == STARTED:
                # update params
                pass
            
            # *text_30* updates
            
            # if text_30 is starting this frame...
            if text_30.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_30.frameNStart = frameN  # exact frame index
                text_30.tStart = t  # local t and not account for scr refresh
                text_30.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_30, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_30.started')
                # update status
                text_30.status = STARTED
                text_30.setAutoDraw(True)
            
            # if text_30 is active this frame...
            if text_30.status == STARTED:
                # update params
                pass
            
            # *slider_8* updates
            
            # if slider_8 is starting this frame...
            if slider_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_8.frameNStart = frameN  # exact frame index
                slider_8.tStart = t  # local t and not account for scr refresh
                slider_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_8.started')
                # update status
                slider_8.status = STARTED
                slider_8.setAutoDraw(True)
            
            # if slider_8 is active this frame...
            if slider_8.status == STARTED:
                # update params
                pass
            # *button_3* updates
            
            # if button_3 is starting this frame...
            if button_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button_3.frameNStart = frameN  # exact frame index
                button_3.tStart = t  # local t and not account for scr refresh
                button_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_3.started')
                # update status
                button_3.status = STARTED
                button_3.setAutoDraw(True)
            
            # if button_3 is active this frame...
            if button_3.status == STARTED:
                # update params
                pass
                # check whether button_3 has been pressed
                if button_3.isClicked:
                    if not button_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_3.timesOn.append(button_3.buttonClock.getTime())
                        button_3.timesOff.append(button_3.buttonClock.getTime())
                    elif len(button_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_3.timesOff[-1] = button_3.buttonClock.getTime()
                    if not button_3.wasClicked:
                        # end routine when button_3 is clicked
                        continueRoutine = False
                    if not button_3.wasClicked:
                        # run callback code when button_3 is clicked
                        pass
            # take note of whether button_3 was clicked, so that next frame we know if clicks are new
            button_3.wasClicked = button_3.isClicked and button_3.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Participant_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Participant_feedback" ---
        for thisComponent in Participant_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Participant_feedback.stopped', globalClock.getTime(format='float'))
        trials_14.addData('slider_6.response', slider_6.getRating())
        trials_14.addData('slider_6.rt', slider_6.getRT())
        trials_14.addData('slider_7.response', slider_7.getRating())
        trials_14.addData('slider_7.rt', slider_7.getRT())
        trials_14.addData('slider_8.response', slider_8.getRating())
        trials_14.addData('slider_8.rt', slider_8.getRT())
        trials_14.addData('button_3.numClicks', button_3.numClicks)
        if button_3.numClicks:
           trials_14.addData('button_3.timesOn', button_3.timesOn)
           trials_14.addData('button_3.timesOff', button_3.timesOff)
        else:
           trials_14.addData('button_3.timesOn', "")
           trials_14.addData('button_3.timesOff', "")
        # the Routine "Participant_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_14'
    
    
    # --- Prepare to start Routine "Instruction_Policyst" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_Policyst.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from code_10
    outlet.push_sample(x=[100])  # Push event marker. Start experiment
    
    # create starting attributes for key_resp_6
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    # keep track of which components have finished
    Instruction_PolicystComponents = [text_22, key_resp_6]
    for thisComponent in Instruction_PolicystComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction_Policyst" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_22* updates
        
        # if text_22 is starting this frame...
        if text_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_22.frameNStart = frameN  # exact frame index
            text_22.tStart = t  # local t and not account for scr refresh
            text_22.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_22, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_22.started')
            # update status
            text_22.status = STARTED
            text_22.setAutoDraw(True)
        
        # if text_22 is active this frame...
        if text_22.status == STARTED:
            # update params
            pass
        
        # *key_resp_6* updates
        waitOnFlip = False
        
        # if key_resp_6 is starting this frame...
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_6.started')
            # update status
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction_PolicystComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction_Policyst" ---
    for thisComponent in Instruction_PolicystComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruction_Policyst.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    thisExp.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        thisExp.addData('key_resp_6.rt', key_resp_6.rt)
        thisExp.addData('key_resp_6.duration', key_resp_6.duration)
    thisExp.nextEntry()
    # the Routine "Instruction_Policyst" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_4 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('statement.xlsx'),
        seed=None, name='trials_4')
    thisExp.addLoop(trials_4)  # add the loop to the experiment
    thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
    if thisTrial_4 != None:
        for paramName in thisTrial_4:
            globals()[paramName] = thisTrial_4[paramName]
    
    for thisTrial_4 in trials_4:
        currentLoop = trials_4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        
        # --- Prepare to start Routine "FixationCross" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('FixationCross.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        FixationCrossComponents = [polygon]
        for thisComponent in FixationCrossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FixationCross" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FixationCrossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FixationCross" ---
        for thisComponent in FixationCrossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('FixationCross.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "P1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('P1.started', globalClock.getTime(format='float'))
        text_58.setText(Statement)
        # Run 'Begin Routine' code from code_35
        outlet.push_sample(x=[7])  # Push event marker. Start experiment
        
        # keep track of which components have finished
        P1Components = [text_58]
        for thisComponent in P1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "P1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_58* updates
            
            # if text_58 is starting this frame...
            if text_58.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_58.frameNStart = frameN  # exact frame index
                text_58.tStart = t  # local t and not account for scr refresh
                text_58.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_58, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_58.started')
                # update status
                text_58.status = STARTED
                text_58.setAutoDraw(True)
            
            # if text_58 is active this frame...
            if text_58.status == STARTED:
                # update params
                pass
            
            # if text_58 is stopping this frame...
            if text_58.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_58.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_58.tStop = t  # not accounting for scr refresh
                    text_58.tStopRefresh = tThisFlipGlobal  # on global time
                    text_58.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_58.stopped')
                    # update status
                    text_58.status = FINISHED
                    text_58.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in P1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "P1" ---
        for thisComponent in P1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('P1.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "trial5" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial5.started', globalClock.getTime(format='float'))
        text_13.setText(Statement)
        text_14.setText('Do you think the statement is true or flase?')
        slider.reset()
        text_15.setText('How confident are you in your answer?')
        slider_2.reset()
        # keep track of which components have finished
        trial5Components = [text_13, text_14, slider, text_15, slider_2]
        for thisComponent in trial5Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial5" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_13* updates
            
            # if text_13 is starting this frame...
            if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_13.frameNStart = frameN  # exact frame index
                text_13.tStart = t  # local t and not account for scr refresh
                text_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_13.started')
                # update status
                text_13.status = STARTED
                text_13.setAutoDraw(True)
            
            # if text_13 is active this frame...
            if text_13.status == STARTED:
                # update params
                pass
            
            # *text_14* updates
            
            # if text_14 is starting this frame...
            if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_14.frameNStart = frameN  # exact frame index
                text_14.tStart = t  # local t and not account for scr refresh
                text_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_14.started')
                # update status
                text_14.status = STARTED
                text_14.setAutoDraw(True)
            
            # if text_14 is active this frame...
            if text_14.status == STARTED:
                # update params
                pass
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # *text_15* updates
            
            # if text_15 is starting this frame...
            if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_15.frameNStart = frameN  # exact frame index
                text_15.tStart = t  # local t and not account for scr refresh
                text_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_15.started')
                # update status
                text_15.status = STARTED
                text_15.setAutoDraw(True)
            
            # if text_15 is active this frame...
            if text_15.status == STARTED:
                # update params
                pass
            
            # *slider_2* updates
            
            # if slider_2 is starting this frame...
            if slider_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_2.frameNStart = frameN  # exact frame index
                slider_2.tStart = t  # local t and not account for scr refresh
                slider_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_2.started')
                # update status
                slider_2.status = STARTED
                slider_2.setAutoDraw(True)
            
            # if slider_2 is active this frame...
            if slider_2.status == STARTED:
                # update params
                pass
            
            # Check slider_2 for response to end Routine
            if slider_2.getRating() is not None and slider_2.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial5Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial5" ---
        for thisComponent in trial5Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial5.stopped', globalClock.getTime(format='float'))
        trials_4.addData('slider.response', slider.getRating())
        trials_4.addData('slider.rt', slider.getRT())
        trials_4.addData('slider_2.response', slider_2.getRating())
        trials_4.addData('slider_2.rt', slider_2.getRT())
        # the Routine "trial5" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime(format='float'))
        text_59.setText('')
        # keep track of which components have finished
        break_2Components = [text_59]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_59* updates
            
            # if text_59 is starting this frame...
            if text_59.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_59.frameNStart = frameN  # exact frame index
                text_59.tStart = t  # local t and not account for scr refresh
                text_59.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_59, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_59.started')
                # update status
                text_59.status = STARTED
                text_59.setAutoDraw(True)
            
            # if text_59 is active this frame...
            if text_59.status == STARTED:
                # update params
                pass
            
            # if text_59 is stopping this frame...
            if text_59.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_59.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_59.tStop = t  # not accounting for scr refresh
                    text_59.tStopRefresh = tThisFlipGlobal  # on global time
                    text_59.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_59.stopped')
                    # update status
                    text_59.status = FINISHED
                    text_59.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "P2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('P2.started', globalClock.getTime(format='float'))
        text_62.setText('Community Notes')
        text_60.setText(communityfeedback)
        # Run 'Begin Routine' code from code_36
        outlet.push_sample(x=[7])  # Push event marker. Start experiment
        
        # keep track of which components have finished
        P2Components = [text_62, text_60]
        for thisComponent in P2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "P2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_62* updates
            
            # if text_62 is starting this frame...
            if text_62.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_62.frameNStart = frameN  # exact frame index
                text_62.tStart = t  # local t and not account for scr refresh
                text_62.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_62, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_62.started')
                # update status
                text_62.status = STARTED
                text_62.setAutoDraw(True)
            
            # if text_62 is active this frame...
            if text_62.status == STARTED:
                # update params
                pass
            
            # if text_62 is stopping this frame...
            if text_62.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_62.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    text_62.tStop = t  # not accounting for scr refresh
                    text_62.tStopRefresh = tThisFlipGlobal  # on global time
                    text_62.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_62.stopped')
                    # update status
                    text_62.status = FINISHED
                    text_62.setAutoDraw(False)
            
            # *text_60* updates
            
            # if text_60 is starting this frame...
            if text_60.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_60.frameNStart = frameN  # exact frame index
                text_60.tStart = t  # local t and not account for scr refresh
                text_60.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_60, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_60.started')
                # update status
                text_60.status = STARTED
                text_60.setAutoDraw(True)
            
            # if text_60 is active this frame...
            if text_60.status == STARTED:
                # update params
                pass
            
            # if text_60 is stopping this frame...
            if text_60.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_60.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    text_60.tStop = t  # not accounting for scr refresh
                    text_60.tStopRefresh = tThisFlipGlobal  # on global time
                    text_60.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_60.stopped')
                    # update status
                    text_60.status = FINISHED
                    text_60.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in P2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "P2" ---
        for thisComponent in P2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('P2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "Revision" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Revision.started', globalClock.getTime(format='float'))
        text_37.setText(Statement)
        text_17.setText('Would you like to revise your estimate: Do you think the statement is true or false?')
        slider_3.reset()
        text_18.setText('How confident are you in your answer?')
        slider_4.reset()
        text_19.setText('How knowledgable are you on the topic?')
        slider_5.reset()
        button.setText('Next')
        # reset button to account for continued clicks & clear times on/off
        button.reset()
        # keep track of which components have finished
        RevisionComponents = [text_37, text_17, slider_3, text_18, slider_4, text_19, slider_5, button]
        for thisComponent in RevisionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Revision" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_37* updates
            
            # if text_37 is starting this frame...
            if text_37.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_37.frameNStart = frameN  # exact frame index
                text_37.tStart = t  # local t and not account for scr refresh
                text_37.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_37, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_37.started')
                # update status
                text_37.status = STARTED
                text_37.setAutoDraw(True)
            
            # if text_37 is active this frame...
            if text_37.status == STARTED:
                # update params
                pass
            
            # *text_17* updates
            
            # if text_17 is starting this frame...
            if text_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_17.frameNStart = frameN  # exact frame index
                text_17.tStart = t  # local t and not account for scr refresh
                text_17.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_17, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_17.started')
                # update status
                text_17.status = STARTED
                text_17.setAutoDraw(True)
            
            # if text_17 is active this frame...
            if text_17.status == STARTED:
                # update params
                pass
            
            # *slider_3* updates
            
            # if slider_3 is starting this frame...
            if slider_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_3.frameNStart = frameN  # exact frame index
                slider_3.tStart = t  # local t and not account for scr refresh
                slider_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_3.started')
                # update status
                slider_3.status = STARTED
                slider_3.setAutoDraw(True)
            
            # if slider_3 is active this frame...
            if slider_3.status == STARTED:
                # update params
                pass
            
            # *text_18* updates
            
            # if text_18 is starting this frame...
            if text_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_18.frameNStart = frameN  # exact frame index
                text_18.tStart = t  # local t and not account for scr refresh
                text_18.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_18, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_18.started')
                # update status
                text_18.status = STARTED
                text_18.setAutoDraw(True)
            
            # if text_18 is active this frame...
            if text_18.status == STARTED:
                # update params
                pass
            
            # *slider_4* updates
            
            # if slider_4 is starting this frame...
            if slider_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_4.frameNStart = frameN  # exact frame index
                slider_4.tStart = t  # local t and not account for scr refresh
                slider_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_4.started')
                # update status
                slider_4.status = STARTED
                slider_4.setAutoDraw(True)
            
            # if slider_4 is active this frame...
            if slider_4.status == STARTED:
                # update params
                pass
            
            # *text_19* updates
            
            # if text_19 is starting this frame...
            if text_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_19.frameNStart = frameN  # exact frame index
                text_19.tStart = t  # local t and not account for scr refresh
                text_19.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_19, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_19.started')
                # update status
                text_19.status = STARTED
                text_19.setAutoDraw(True)
            
            # if text_19 is active this frame...
            if text_19.status == STARTED:
                # update params
                pass
            
            # *slider_5* updates
            
            # if slider_5 is starting this frame...
            if slider_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_5.frameNStart = frameN  # exact frame index
                slider_5.tStart = t  # local t and not account for scr refresh
                slider_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_5.started')
                # update status
                slider_5.status = STARTED
                slider_5.setAutoDraw(True)
            
            # if slider_5 is active this frame...
            if slider_5.status == STARTED:
                # update params
                pass
            # *button* updates
            
            # if button is starting this frame...
            if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button.frameNStart = frameN  # exact frame index
                button.tStart = t  # local t and not account for scr refresh
                button.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button.started')
                # update status
                button.status = STARTED
                button.setAutoDraw(True)
            
            # if button is active this frame...
            if button.status == STARTED:
                # update params
                pass
                # check whether button has been pressed
                if button.isClicked:
                    if not button.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button.timesOn.append(button.buttonClock.getTime())
                        button.timesOff.append(button.buttonClock.getTime())
                    elif len(button.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button.timesOff[-1] = button.buttonClock.getTime()
                    if not button.wasClicked:
                        # end routine when button is clicked
                        continueRoutine = False
                    if not button.wasClicked:
                        # run callback code when button is clicked
                        pass
            # take note of whether button was clicked, so that next frame we know if clicks are new
            button.wasClicked = button.isClicked and button.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in RevisionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Revision" ---
        for thisComponent in RevisionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Revision.stopped', globalClock.getTime(format='float'))
        trials_4.addData('slider_3.response', slider_3.getRating())
        trials_4.addData('slider_3.rt', slider_3.getRT())
        trials_4.addData('slider_4.response', slider_4.getRating())
        trials_4.addData('slider_4.rt', slider_4.getRT())
        trials_4.addData('slider_5.response', slider_5.getRating())
        trials_4.addData('slider_5.rt', slider_5.getRT())
        trials_4.addData('button.numClicks', button.numClicks)
        if button.numClicks:
           trials_4.addData('button.timesOn', button.timesOn)
           trials_4.addData('button.timesOff', button.timesOff)
        else:
           trials_4.addData('button.timesOn', "")
           trials_4.addData('button.timesOff', "")
        # the Routine "Revision" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime(format='float'))
        text_59.setText('')
        # keep track of which components have finished
        break_2Components = [text_59]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_59* updates
            
            # if text_59 is starting this frame...
            if text_59.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_59.frameNStart = frameN  # exact frame index
                text_59.tStart = t  # local t and not account for scr refresh
                text_59.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_59, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_59.started')
                # update status
                text_59.status = STARTED
                text_59.setAutoDraw(True)
            
            # if text_59 is active this frame...
            if text_59.status == STARTED:
                # update params
                pass
            
            # if text_59 is stopping this frame...
            if text_59.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_59.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_59.tStop = t  # not accounting for scr refresh
                    text_59.tStopRefresh = tThisFlipGlobal  # on global time
                    text_59.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_59.stopped')
                    # update status
                    text_59.status = FINISHED
                    text_59.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "P3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('P3.started', globalClock.getTime(format='float'))
        text_63.setText('The headline you saw was:')
        text_61.setText(feedback)
        # Run 'Begin Routine' code from code_37
        outlet.push_sample(x=[7])  # Push event marker. Start experiment
        
        # keep track of which components have finished
        P3Components = [text_63, text_61]
        for thisComponent in P3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "P3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_63* updates
            
            # if text_63 is starting this frame...
            if text_63.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_63.frameNStart = frameN  # exact frame index
                text_63.tStart = t  # local t and not account for scr refresh
                text_63.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_63, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_63.started')
                # update status
                text_63.status = STARTED
                text_63.setAutoDraw(True)
            
            # if text_63 is active this frame...
            if text_63.status == STARTED:
                # update params
                pass
            
            # if text_63 is stopping this frame...
            if text_63.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_63.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_63.tStop = t  # not accounting for scr refresh
                    text_63.tStopRefresh = tThisFlipGlobal  # on global time
                    text_63.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_63.stopped')
                    # update status
                    text_63.status = FINISHED
                    text_63.setAutoDraw(False)
            
            # *text_61* updates
            
            # if text_61 is starting this frame...
            if text_61.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_61.frameNStart = frameN  # exact frame index
                text_61.tStart = t  # local t and not account for scr refresh
                text_61.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_61, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_61.started')
                # update status
                text_61.status = STARTED
                text_61.setAutoDraw(True)
            
            # if text_61 is active this frame...
            if text_61.status == STARTED:
                # update params
                pass
            
            # if text_61 is stopping this frame...
            if text_61.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_61.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_61.tStop = t  # not accounting for scr refresh
                    text_61.tStopRefresh = tThisFlipGlobal  # on global time
                    text_61.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_61.stopped')
                    # update status
                    text_61.status = FINISHED
                    text_61.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in P3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "P3" ---
        for thisComponent in P3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('P3.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_4'
    
    
    # --- Prepare to start Routine "Beginning_questionnaire" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Beginning_questionnaire.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_10
    key_resp_10.keys = []
    key_resp_10.rt = []
    _key_resp_10_allKeys = []
    # keep track of which components have finished
    Beginning_questionnaireComponents = [text_25, key_resp_10]
    for thisComponent in Beginning_questionnaireComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Beginning_questionnaire" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_25* updates
        
        # if text_25 is starting this frame...
        if text_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_25.frameNStart = frameN  # exact frame index
            text_25.tStart = t  # local t and not account for scr refresh
            text_25.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_25, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_25.started')
            # update status
            text_25.status = STARTED
            text_25.setAutoDraw(True)
        
        # if text_25 is active this frame...
        if text_25.status == STARTED:
            # update params
            pass
        
        # *key_resp_10* updates
        waitOnFlip = False
        
        # if key_resp_10 is starting this frame...
        if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_10.frameNStart = frameN  # exact frame index
            key_resp_10.tStart = t  # local t and not account for scr refresh
            key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_10.started')
            # update status
            key_resp_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_10.getKeys(keyList=['a','b','c','d'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_10_allKeys.extend(theseKeys)
            if len(_key_resp_10_allKeys):
                key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Beginning_questionnaireComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Beginning_questionnaire" ---
    for thisComponent in Beginning_questionnaireComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Beginning_questionnaire.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_10.keys in ['', [], None]:  # No response was made
        key_resp_10.keys = None
    thisExp.addData('key_resp_10.keys',key_resp_10.keys)
    if key_resp_10.keys != None:  # we had a response
        thisExp.addData('key_resp_10.rt', key_resp_10.rt)
        thisExp.addData('key_resp_10.duration', key_resp_10.duration)
    thisExp.nextEntry()
    # the Routine "Beginning_questionnaire" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "EndExperiment" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('EndExperiment.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    EndExperimentComponents = [text_20]
    for thisComponent in EndExperimentComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndExperiment" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_20* updates
        
        # if text_20 is starting this frame...
        if text_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_20.frameNStart = frameN  # exact frame index
            text_20.tStart = t  # local t and not account for scr refresh
            text_20.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_20, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_20.started')
            # update status
            text_20.status = STARTED
            text_20.setAutoDraw(True)
        
        # if text_20 is active this frame...
        if text_20.status == STARTED:
            # update params
            pass
        
        # if text_20 is stopping this frame...
        if text_20.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_20.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_20.tStop = t  # not accounting for scr refresh
                text_20.tStopRefresh = tThisFlipGlobal  # on global time
                text_20.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_20.stopped')
                # update status
                text_20.status = FINISHED
                text_20.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndExperimentComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndExperiment" ---
    for thisComponent in EndExperimentComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('EndExperiment.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
