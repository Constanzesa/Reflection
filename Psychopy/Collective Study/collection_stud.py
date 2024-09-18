#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.1),
    on Tue Apr 23 19:23:14 2024
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
psychopyVersion = '2024.1.1'
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
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
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
        originPath='/Users/constanze/Desktop/Study1/Collective Study/collection_stud.py',
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
            size=[1440, 900], fullscr=_fullScr, screen=0,
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
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
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
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
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
    
    # --- Initialize components for Routine "setrow" ---
    # Run 'Begin Experiment' code from code
    useRows ='0:3'
    
    
    # --- Initialize components for Routine "trial1" ---
    
    # --- Initialize components for Routine "FixationCross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "BaseRatequestion" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "setRows2" ---
    # Run 'Begin Experiment' code from code_2
    useRows2= '0'
    
    # --- Initialize components for Routine "baserateresponse" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Instruction_Syl_2" ---
    text_21 = visual.TextStim(win=win, name='text_21',
        text="In this experiment you will be given four problems. In each case, you will be given a prose passage to read and asked if a certain conclusion may be logically deduced from it. You should answer this question on the assumption that all the information given in the passage is, in fact, true. If you judge that the conclusion necessarily follows from the statements in the passage, you should answer\n\n'yes,' otherwise 'no.'\nYou can start the experiment by pressing the 'space' bar",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    
    # --- Initialize components for Routine "setRow3" ---
    # Run 'Begin Experiment' code from code_3
    useRows3 = '0:3'
    
    # --- Initialize components for Routine "trialcount3" ---
    
    # --- Initialize components for Routine "FixationCross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Syllologistic_TypeA" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='',
        font='Open Sans',
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
    
    # --- Initialize components for Routine "setRow4" ---
    # Run 'Begin Experiment' code from code_4
    useRows4='0:3'
    
    # --- Initialize components for Routine "trialcount4" ---
    
    # --- Initialize components for Routine "FixationCross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "TypeB" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='',
        font='Open Sans',
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
        text="In this experiment you will be presented with a series of questions, one at a time. Each question has one correct answer.\n\nPlease first type the number that you think solves the question. \nIf you want to skip to the next trial please press the 'space' bar.\n\nYou can start the experiment by pressing the 'space' bar\n",
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
    
    # --- Initialize components for Routine "Beginning_questionnaire" ---
    text_25 = visual.TextStim(win=win, name='text_25',
        text='Before we start the experiment, please indicate which of the following best describes your usual political stance?\n\na) Republican\nb) Democratic\nc) Independent\nd) Other\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
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
        text='Do you think the statement is true or false?',
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
    
    # --- Initialize components for Routine "setRow5" ---
    # Run 'Begin Experiment' code from code_5
    useRows5='0'
    
    # --- Initialize components for Routine "trialcount5" ---
    
    # --- Initialize components for Routine "Policy_statement" ---
    Policy_statement_secondtrial = visual.TextStim(win=win, name='Policy_statement_secondtrial',
        text='',
        font='Open Sans',
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
    
    # --- Initialize components for Routine "break_short" ---
    text_12 = visual.TextStim(win=win, name='text_12',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "communitynotes" ---
    text_36 = visual.TextStim(win=win, name='text_36',
        text='',
        font='Open Sans',
        pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_16 = visual.TextStim(win=win, name='text_16',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
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
    
    # --- Initialize components for Routine "break_short" ---
    text_12 = visual.TextStim(win=win, name='text_12',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Feedback" ---
    text_24 = visual.TextStim(win=win, name='text_24',
        text='The headline you saw was:',
        font='Open Sans',
        pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_23 = visual.TextStim(win=win, name='text_23',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
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
    if key_resp_4.status == STARTED:
            # LSL MARKER 
            outlet.push_sample(x=[100])  # Push event marker. Start experiment
    
    thisExp.nextEntry()
    # the Routine "Instruction_BaseRate" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    block = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('random.xlsx'),
        seed=None, name='block')
    thisExp.addLoop(block)  # add the loop to the experiment
    thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in block:
        currentLoop = block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "setrow" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('setrow.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code
        if random == 0:
            useRows = '0:3'
        if random== 1:
            useRows = '3:6'
        if random == 2:
            useRows = '6:9'
        if random== 3:
            useRows = '9:12'
        if random == 4:
            useRows = '12:15'
        if random == 5:
            useRows = '15:18'
        if random == 6:
            useRows = '18:21'
        if random== 7:
            useRows = '21:24'
        
        # keep track of which components have finished
        setrowComponents = []
        for thisComponent in setrowComponents:
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
        
        # --- Run Routine "setrow" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            for thisComponent in setrowComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "setrow" ---
        for thisComponent in setrowComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('setrow.stopped', globalClock.getTime(format='float'))
        # the Routine "setrow" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('stim_number.xlsx', selection=useRows),
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
            
            # --- Prepare to start Routine "trial1" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial1.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            trial1Components = []
            for thisComponent in trial1Components:
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
            
            # --- Run Routine "trial1" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                for thisComponent in trial1Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial1" ---
            for thisComponent in trial1Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial1.stopped', globalClock.getTime(format='float'))
            # the Routine "trial1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials'
        
        
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
        
        # set up handler to look after randomisation of conditions etc
        trials_2 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('BaseRateinfo.xlsx', selection=useRows),
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
            
            # --- Prepare to start Routine "BaseRatequestion" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('BaseRatequestion.started', globalClock.getTime(format='float'))
            text_3.setText(BaseRateinfo)
            # keep track of which components have finished
            BaseRatequestionComponents = [text_3]
            for thisComponent in BaseRatequestionComponents:
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
            
            # --- Run Routine "BaseRatequestion" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_3* updates
                
                # if text_3 is starting this frame...
                if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_3.frameNStart = frameN  # exact frame index
                    text_3.tStart = t  # local t and not account for scr refresh
                    text_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.started')
                    # update status
                    text_3.status = STARTED
                    text_3.setAutoDraw(True)
                
                # if text_3 is active this frame...
                if text_3.status == STARTED:
                    # update params
                    pass
                
                # if text_3 is stopping this frame...
                if text_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_3.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_3.tStop = t  # not accounting for scr refresh
                        text_3.tStopRefresh = tThisFlipGlobal  # on global time
                        text_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_3.stopped')
                        # update status
                        text_3.status = FINISHED
                        text_3.setAutoDraw(False)
                
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
                for thisComponent in BaseRatequestionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "BaseRatequestion" ---
            for thisComponent in BaseRatequestionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('BaseRatequestion.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_2'
        
        
        # --- Prepare to start Routine "setRows2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('setRows2.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_2
        if random == 0:
            useRows2 = '0'
        if random== 1:
            useRows2 = '1'
        if random == 2:
            useRows2 = '2'
        if random== 3:
            useRows2 = '3'
        if random == 4:
            useRows2 = '4'
        if random == 5:
            useRows2 = '5'
        if random == 6:
            useRows2 = '6'
        if random== 7:
            useRows2 = '7'
        
        # keep track of which components have finished
        setRows2Components = []
        for thisComponent in setRows2Components:
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
        
        # --- Run Routine "setRows2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            for thisComponent in setRows2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "setRows2" ---
        for thisComponent in setRows2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('setRows2.stopped', globalClock.getTime(format='float'))
        # the Routine "setRows2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_3 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('baserate_responses.xlsx', selection=useRows2),
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
            
            # --- Prepare to start Routine "baserateresponse" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('baserateresponse.started', globalClock.getTime(format='float'))
            text_4.setText(baserate_response)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            baserateresponseComponents = [text_4, key_resp]
            for thisComponent in baserateresponseComponents:
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
            
            # --- Run Routine "baserateresponse" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_4* updates
                
                # if text_4 is starting this frame...
                if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_4.frameNStart = frameN  # exact frame index
                    text_4.tStart = t  # local t and not account for scr refresh
                    text_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.started')
                    # update status
                    text_4.status = STARTED
                    text_4.setAutoDraw(True)
                
                # if text_4 is active this frame...
                if text_4.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('key_resp.started', t)
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    key_resp.clock.reset()  # now t=0
                    key_resp.clearEvents(eventType='keyboard')
                if key_resp.status == STARTED:
                    theseKeys = key_resp.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # was this correct?
                        if (key_resp.keys == str(key_ab)) or (key_resp.keys == key_ab):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
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
                for thisComponent in baserateresponseComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "baserateresponse" ---
            for thisComponent in baserateresponseComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('baserateresponse.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(key_ab).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_3 (TrialHandler)
            trials_3.addData('key_resp.keys',key_resp.keys)
            trials_3.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                trials_3.addData('key_resp.rt', key_resp.rt)
                trials_3.addData('key_resp.duration', key_resp.duration)
            # Run 'End Routine' code from LSL_baserateresponse
            if key_resp.status == STARTED:
                outlet.push_sample(x=[marker_baserate]) # Push event marker. Baserateresponse ==1
            # the Routine "baserateresponse" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_3'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'block'
    
    
    # --- Prepare to start Routine "Instruction_Syl_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_Syl_2.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from LSL_Start
    
    
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
    if key_resp_4.status == STARTED:
    
            # LSL MARKER 
    
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
    block1 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('random2.xlsx'),
        seed=None, name='block1')
    thisExp.addLoop(block1)  # add the loop to the experiment
    thisBlock1 = block1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock1.rgb)
    if thisBlock1 != None:
        for paramName in thisBlock1:
            globals()[paramName] = thisBlock1[paramName]
    
    for thisBlock1 in block1:
        currentLoop = block1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock1.rgb)
        if thisBlock1 != None:
            for paramName in thisBlock1:
                globals()[paramName] = thisBlock1[paramName]
        
        # --- Prepare to start Routine "setRow3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('setRow3.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_3
        if random2 == 0:
            useRows3 = '0:3'
        if random2== 1:
            useRows3 = '3:6'
        if random2 == 2:
            useRows3 = '6:9'
        if random2 == 3:
            useRows3 = '9:12'
        if random2 == 4:
            useRows3 = '12:15'
        if random2 == 5:
            useRows3 = '15:18'
        if random2 == 6:
            useRows3 = '18:21'
        if random2 == 7:
            useRows3 = '21:24'
        # keep track of which components have finished
        setRow3Components = []
        for thisComponent in setRow3Components:
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
        
        # --- Run Routine "setRow3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            for thisComponent in setRow3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "setRow3" ---
        for thisComponent in setRow3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('setRow3.stopped', globalClock.getTime(format='float'))
        # the Routine "setRow3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_4 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('trial_number.xlsx', selection=useRows3),
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
            
            # --- Prepare to start Routine "trialcount3" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trialcount3.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            trialcount3Components = []
            for thisComponent in trialcount3Components:
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
            
            # --- Run Routine "trialcount3" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                for thisComponent in trialcount3Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trialcount3" ---
            for thisComponent in trialcount3Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trialcount3.stopped', globalClock.getTime(format='float'))
            # the Routine "trialcount3" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_4'
        
        
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
        
        # set up handler to look after randomisation of conditions etc
        trials_5 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('SyllologisticTask.xlsx', selection=useRows3),
            seed=None, name='trials_5')
        thisExp.addLoop(trials_5)  # add the loop to the experiment
        thisTrial_5 = trials_5.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
        if thisTrial_5 != None:
            for paramName in thisTrial_5:
                globals()[paramName] = thisTrial_5[paramName]
        
        for thisTrial_5 in trials_5:
            currentLoop = trials_5
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
            if thisTrial_5 != None:
                for paramName in thisTrial_5:
                    globals()[paramName] = thisTrial_5[paramName]
            
            # --- Prepare to start Routine "Syllologistic_TypeA" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Syllologistic_TypeA.started', globalClock.getTime(format='float'))
            text_5.setText(Type_A)
            # keep track of which components have finished
            Syllologistic_TypeAComponents = [text_5]
            for thisComponent in Syllologistic_TypeAComponents:
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
            
            # --- Run Routine "Syllologistic_TypeA" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_5* updates
                
                # if text_5 is starting this frame...
                if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_5.frameNStart = frameN  # exact frame index
                    text_5.tStart = t  # local t and not account for scr refresh
                    text_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_5.started')
                    # update status
                    text_5.status = STARTED
                    text_5.setAutoDraw(True)
                
                # if text_5 is active this frame...
                if text_5.status == STARTED:
                    # update params
                    pass
                
                # if text_5 is stopping this frame...
                if text_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_5.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_5.tStop = t  # not accounting for scr refresh
                        text_5.tStopRefresh = tThisFlipGlobal  # on global time
                        text_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_5.stopped')
                        # update status
                        text_5.status = FINISHED
                        text_5.setAutoDraw(False)
                
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
                for thisComponent in Syllologistic_TypeAComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Syllologistic_TypeA" ---
            for thisComponent in Syllologistic_TypeAComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Syllologistic_TypeA.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_5'
        
        
        # set up handler to look after randomisation of conditions etc
        marker_2 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('marker2_SyllogistiTask.xlsx'),
            seed=None, name='marker_2')
        thisExp.addLoop(marker_2)  # add the loop to the experiment
        thisMarker_2 = marker_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMarker_2.rgb)
        if thisMarker_2 != None:
            for paramName in thisMarker_2:
                globals()[paramName] = thisMarker_2[paramName]
        
        for thisMarker_2 in marker_2:
            currentLoop = marker_2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisMarker_2.rgb)
            if thisMarker_2 != None:
                for paramName in thisMarker_2:
                    globals()[paramName] = thisMarker_2[paramName]
            
            # --- Prepare to start Routine "question_logicalconclu" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('question_logicalconclu.started', globalClock.getTime(format='float'))
            text_6.setText('Does the conclusion follow logically?\n\na) yes\nb) no')
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
            # Run 'End Routine' code from code_7
            if key_resp_2.status == STARTED:
                # LSL MARKER
                outlet.push_sample(x=[marker2_Syllogistic])  # Push event marker. syllogistic_reasoning==1
            
            
            # check responses
            if key_resp_2.keys in ['', [], None]:  # No response was made
                key_resp_2.keys = None
                # was no response the correct answer?!
                if str(key_ab1).lower() == 'none':
                   key_resp_2.corr = 1;  # correct non-response
                else:
                   key_resp_2.corr = 0;  # failed to respond (incorrectly)
            # store data for marker_2 (TrialHandler)
            marker_2.addData('key_resp_2.keys',key_resp_2.keys)
            marker_2.addData('key_resp_2.corr', key_resp_2.corr)
            if key_resp_2.keys != None:  # we had a response
                marker_2.addData('key_resp_2.rt', key_resp_2.rt)
                marker_2.addData('key_resp_2.duration', key_resp_2.duration)
            # the Routine "question_logicalconclu" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'marker_2'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'block1'
    
    
    # set up handler to look after randomisation of conditions etc
    block3 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('random2.xlsx'),
        seed=None, name='block3')
    thisExp.addLoop(block3)  # add the loop to the experiment
    thisBlock3 = block3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock3.rgb)
    if thisBlock3 != None:
        for paramName in thisBlock3:
            globals()[paramName] = thisBlock3[paramName]
    
    for thisBlock3 in block3:
        currentLoop = block3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock3.rgb)
        if thisBlock3 != None:
            for paramName in thisBlock3:
                globals()[paramName] = thisBlock3[paramName]
        
        # --- Prepare to start Routine "setRow4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('setRow4.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_4
        if random2 == 0:
            useRows4 = '0:3'
        if random2== 1:
            useRows4 = '3:6'
        if random2 == 2:
            useRows4 = '6:9'
        if random2 == 3:
            useRows4 = '9:12'
        if random2 == 4:
            useRows4 = '12:15'
        if random2 == 5:
            useRows4 = '15:18'
        if random2 == 6:
            useRows4 = '18:21'
        if random2 == 7:
            useRows4 = '21:24'
        # keep track of which components have finished
        setRow4Components = []
        for thisComponent in setRow4Components:
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
        
        # --- Run Routine "setRow4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            for thisComponent in setRow4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "setRow4" ---
        for thisComponent in setRow4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('setRow4.stopped', globalClock.getTime(format='float'))
        # the Routine "setRow4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_6 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('trial_number.xlsx', selection=useRows4),
            seed=None, name='trials_6')
        thisExp.addLoop(trials_6)  # add the loop to the experiment
        thisTrial_6 = trials_6.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_6.rgb)
        if thisTrial_6 != None:
            for paramName in thisTrial_6:
                globals()[paramName] = thisTrial_6[paramName]
        
        for thisTrial_6 in trials_6:
            currentLoop = trials_6
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_6.rgb)
            if thisTrial_6 != None:
                for paramName in thisTrial_6:
                    globals()[paramName] = thisTrial_6[paramName]
            
            # --- Prepare to start Routine "trialcount4" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trialcount4.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            trialcount4Components = []
            for thisComponent in trialcount4Components:
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
            
            # --- Run Routine "trialcount4" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                for thisComponent in trialcount4Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trialcount4" ---
            for thisComponent in trialcount4Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trialcount4.stopped', globalClock.getTime(format='float'))
            # the Routine "trialcount4" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_6'
        
        
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
        
        # set up handler to look after randomisation of conditions etc
        trials_7 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('TypeB.xlsx', selection=useRows4),
            seed=None, name='trials_7')
        thisExp.addLoop(trials_7)  # add the loop to the experiment
        thisTrial_7 = trials_7.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_7.rgb)
        if thisTrial_7 != None:
            for paramName in thisTrial_7:
                globals()[paramName] = thisTrial_7[paramName]
        
        for thisTrial_7 in trials_7:
            currentLoop = trials_7
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_7.rgb)
            if thisTrial_7 != None:
                for paramName in thisTrial_7:
                    globals()[paramName] = thisTrial_7[paramName]
            
            # --- Prepare to start Routine "TypeB" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('TypeB.started', globalClock.getTime(format='float'))
            text_8.setText(Type_B)
            # keep track of which components have finished
            TypeBComponents = [text_8]
            for thisComponent in TypeBComponents:
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
            
            # --- Run Routine "TypeB" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_8* updates
                
                # if text_8 is starting this frame...
                if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_8.frameNStart = frameN  # exact frame index
                    text_8.tStart = t  # local t and not account for scr refresh
                    text_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.started')
                    # update status
                    text_8.status = STARTED
                    text_8.setAutoDraw(True)
                
                # if text_8 is active this frame...
                if text_8.status == STARTED:
                    # update params
                    pass
                
                # if text_8 is stopping this frame...
                if text_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_8.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_8.tStop = t  # not accounting for scr refresh
                        text_8.tStopRefresh = tThisFlipGlobal  # on global time
                        text_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_8.stopped')
                        # update status
                        text_8.status = FINISHED
                        text_8.setAutoDraw(False)
                
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
                for thisComponent in TypeBComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "TypeB" ---
            for thisComponent in TypeBComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('TypeB.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_7'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_13 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('marker2_SyllogistiTask.xlsx'),
            seed=None, name='trials_13')
        thisExp.addLoop(trials_13)  # add the loop to the experiment
        thisTrial_13 = trials_13.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_13.rgb)
        if thisTrial_13 != None:
            for paramName in thisTrial_13:
                globals()[paramName] = thisTrial_13[paramName]
        
        for thisTrial_13 in trials_13:
            currentLoop = trials_13
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_13.rgb)
            if thisTrial_13 != None:
                for paramName in thisTrial_13:
                    globals()[paramName] = thisTrial_13[paramName]
            
            # --- Prepare to start Routine "question_locical" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('question_locical.started', globalClock.getTime(format='float'))
            text_35.setText('Does the conclusion follow logically?\n\na) yes\nb) no')
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
            # Run 'End Routine' code from code_16
            if key_resp_2.status == STARTED:
                # LSL MARKER
                outlet.push_sample(x=[marker2_Syllogistic])  # Push event marker. syllogistic_reasoning==1
            
            
            # check responses
            if key_resp_11.keys in ['', [], None]:  # No response was made
                key_resp_11.keys = None
                # was no response the correct answer?!
                if str('key_ab2').lower() == 'none':
                   key_resp_11.corr = 1;  # correct non-response
                else:
                   key_resp_11.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_13 (TrialHandler)
            trials_13.addData('key_resp_11.keys',key_resp_11.keys)
            trials_13.addData('key_resp_11.corr', key_resp_11.corr)
            if key_resp_11.keys != None:  # we had a response
                trials_13.addData('key_resp_11.rt', key_resp_11.rt)
                trials_13.addData('key_resp_11.duration', key_resp_11.duration)
            # the Routine "question_locical" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_13'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'block3'
    
    
    # --- Prepare to start Routine "Instruction_CRT" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_CRT.started', globalClock.getTime(format='float'))
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
        text_11.setText(CRT)
        textbox.reset()
        textbox.setText('')
        textbox.setPlaceholder('')
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
        # Run 'End Routine' code from LSL_CRT
        if key_resp_3.status == STARTED:
            outlet.push_sample(x=[marker3_CRT])
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
    
    # --- Prepare to start Routine "Beginning_questionnaire" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Beginning_questionnaire.started', globalClock.getTime(format='float'))
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
            outlet.push_sample(x=[marker4])  # Push event marker. Baseline==1
        
        
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
    # Run 'End Routine' code from code_10
    if key_resp_4.status == STARTED:
            # LSL MARKER 
            outlet.push_sample(x=[100])  # Push event marker. Start experiment
    
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
    block4 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('random7.xlsx'),
        seed=None, name='block4')
    thisExp.addLoop(block4)  # add the loop to the experiment
    thisBlock4 = block4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock4.rgb)
    if thisBlock4 != None:
        for paramName in thisBlock4:
            globals()[paramName] = thisBlock4[paramName]
    
    for thisBlock4 in block4:
        currentLoop = block4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock4.rgb)
        if thisBlock4 != None:
            for paramName in thisBlock4:
                globals()[paramName] = thisBlock4[paramName]
        
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
        
        # --- Prepare to start Routine "setRow5" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('setRow5.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_5
        if  random3 == 0:
            useRows5= '0'
        if random3== 1:
            useRows5= '1'
        if  random3 == 2:
            useRows5= '2'
        if  random3== 3:
            useRows5= '3'
        if  random3 == 4:
            useRows5= '4'
        if random3 == 5:
            useRows5= '5'
        if  random3 == 6:
            useRows5= '6'
        if  random3 == 7:
            useRows5= '7'
        if  random3 == 8:
            useRows5= '8'
        if  random3 == 9:
            useRows5= '9'
        if random3 == 10:
            useRows5= '10'
        if  random3 == 11:
            useRows5= '11'
        if  random3 == 12:
            useRows5= '12'
        if  random3 == 13:
            useRows5= '13'
        if  block4.thisRepN == 14:
            useRows5= '14'  
        if  block4.thisRepN == 15:
            useRows5= '15'
        if  block4.thisRepN == 16:
            useRows5= '16'
        if  block4.thisRepN == 17:
            useRows5= '17'    
        if  block4.thisRepN == 18:
            useRows5= '18'   
        if  block4.thisRepN == 17:
            useRows5= '18'   
        if  block4.thisRepN == 19:
            useRows5= '19'
        if  block4.thisRepN == 20:
            useRows5= '20'   
        if  block4.thisRepN == 21:
            useRows5= '21'   
        if  block4.thisRepN == 22:
            useRows5= '22'   
        if  block4.thisRepN == 23:
            useRows5= '23'   
        if  block4.thisRepN == 24:
            useRows5= '24'   
        if  random3 == 17:
            useRows5= '17'   
        if  random3== 17:
            useRows5= '17'   
        if  random3== 17:
            useRows5= '17'   
        
        # keep track of which components have finished
        setRow5Components = []
        for thisComponent in setRow5Components:
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
        
        # --- Run Routine "setRow5" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            for thisComponent in setRow5Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "setRow5" ---
        for thisComponent in setRow5Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('setRow5.stopped', globalClock.getTime(format='float'))
        # the Routine "setRow5" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_9 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('count.xlsx', selection=useRows5),
            seed=None, name='trials_9')
        thisExp.addLoop(trials_9)  # add the loop to the experiment
        thisTrial_9 = trials_9.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_9.rgb)
        if thisTrial_9 != None:
            for paramName in thisTrial_9:
                globals()[paramName] = thisTrial_9[paramName]
        
        for thisTrial_9 in trials_9:
            currentLoop = trials_9
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_9.rgb)
            if thisTrial_9 != None:
                for paramName in thisTrial_9:
                    globals()[paramName] = thisTrial_9[paramName]
            
            # --- Prepare to start Routine "trialcount5" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trialcount5.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            trialcount5Components = []
            for thisComponent in trialcount5Components:
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
            
            # --- Run Routine "trialcount5" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
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
                for thisComponent in trialcount5Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trialcount5" ---
            for thisComponent in trialcount5Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trialcount5.stopped', globalClock.getTime(format='float'))
            # the Routine "trialcount5" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_9'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_10 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('statement.xlsx', selection=useRows5),
            seed=None, name='trials_10')
        thisExp.addLoop(trials_10)  # add the loop to the experiment
        thisTrial_10 = trials_10.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_10.rgb)
        if thisTrial_10 != None:
            for paramName in thisTrial_10:
                globals()[paramName] = thisTrial_10[paramName]
        
        for thisTrial_10 in trials_10:
            currentLoop = trials_10
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_10.rgb)
            if thisTrial_10 != None:
                for paramName in thisTrial_10:
                    globals()[paramName] = thisTrial_10[paramName]
            
            # --- Prepare to start Routine "Policy_statement" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Policy_statement.started', globalClock.getTime(format='float'))
            Policy_statement_secondtrial.setText(Statement)
            # keep track of which components have finished
            Policy_statementComponents = [Policy_statement_secondtrial]
            for thisComponent in Policy_statementComponents:
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
            
            # --- Run Routine "Policy_statement" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Policy_statement_secondtrial* updates
                
                # if Policy_statement_secondtrial is starting this frame...
                if Policy_statement_secondtrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Policy_statement_secondtrial.frameNStart = frameN  # exact frame index
                    Policy_statement_secondtrial.tStart = t  # local t and not account for scr refresh
                    Policy_statement_secondtrial.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Policy_statement_secondtrial, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Policy_statement_secondtrial.started')
                    # update status
                    Policy_statement_secondtrial.status = STARTED
                    Policy_statement_secondtrial.setAutoDraw(True)
                
                # if Policy_statement_secondtrial is active this frame...
                if Policy_statement_secondtrial.status == STARTED:
                    # update params
                    pass
                
                # if Policy_statement_secondtrial is stopping this frame...
                if Policy_statement_secondtrial.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Policy_statement_secondtrial.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        Policy_statement_secondtrial.tStop = t  # not accounting for scr refresh
                        Policy_statement_secondtrial.tStopRefresh = tThisFlipGlobal  # on global time
                        Policy_statement_secondtrial.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Policy_statement_secondtrial.stopped')
                        # update status
                        Policy_statement_secondtrial.status = FINISHED
                        Policy_statement_secondtrial.setAutoDraw(False)
                
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
                for thisComponent in Policy_statementComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Policy_statement" ---
            for thisComponent in Policy_statementComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Policy_statement.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from code_15
            outlet.push_sample(x=[marker6_PolicyStatement2])
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
            trials_10.addData('slider.response', slider.getRating())
            trials_10.addData('slider.rt', slider.getRT())
            trials_10.addData('slider_2.response', slider_2.getRating())
            trials_10.addData('slider_2.rt', slider_2.getRT())
            # the Routine "trial5" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "break_short" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('break_short.started', globalClock.getTime(format='float'))
            text_12.setText('')
            # keep track of which components have finished
            break_shortComponents = [text_12]
            for thisComponent in break_shortComponents:
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
            
            # --- Run Routine "break_short" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_12* updates
                
                # if text_12 is starting this frame...
                if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_12.frameNStart = frameN  # exact frame index
                    text_12.tStart = t  # local t and not account for scr refresh
                    text_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_12.started')
                    # update status
                    text_12.status = STARTED
                    text_12.setAutoDraw(True)
                
                # if text_12 is active this frame...
                if text_12.status == STARTED:
                    # update params
                    pass
                
                # if text_12 is stopping this frame...
                if text_12.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_12.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text_12.tStop = t  # not accounting for scr refresh
                        text_12.tStopRefresh = tThisFlipGlobal  # on global time
                        text_12.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_12.stopped')
                        # update status
                        text_12.status = FINISHED
                        text_12.setAutoDraw(False)
                
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
                for thisComponent in break_shortComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "break_short" ---
            for thisComponent in break_shortComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('break_short.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_10'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_11 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('communitynotes.xlsx', selection=useRows5),
            seed=None, name='trials_11')
        thisExp.addLoop(trials_11)  # add the loop to the experiment
        thisTrial_11 = trials_11.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_11.rgb)
        if thisTrial_11 != None:
            for paramName in thisTrial_11:
                globals()[paramName] = thisTrial_11[paramName]
        
        for thisTrial_11 in trials_11:
            currentLoop = trials_11
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_11.rgb)
            if thisTrial_11 != None:
                for paramName in thisTrial_11:
                    globals()[paramName] = thisTrial_11[paramName]
            
            # --- Prepare to start Routine "communitynotes" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('communitynotes.started', globalClock.getTime(format='float'))
            text_36.setText('Community Notes')
            text_16.setText(communityfeedback)
            # keep track of which components have finished
            communitynotesComponents = [text_36, text_16]
            for thisComponent in communitynotesComponents:
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
            
            # --- Run Routine "communitynotes" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_36* updates
                
                # if text_36 is starting this frame...
                if text_36.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_36.frameNStart = frameN  # exact frame index
                    text_36.tStart = t  # local t and not account for scr refresh
                    text_36.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_36, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_36.started')
                    # update status
                    text_36.status = STARTED
                    text_36.setAutoDraw(True)
                
                # if text_36 is active this frame...
                if text_36.status == STARTED:
                    # update params
                    pass
                
                # if text_36 is stopping this frame...
                if text_36.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_36.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        text_36.tStop = t  # not accounting for scr refresh
                        text_36.tStopRefresh = tThisFlipGlobal  # on global time
                        text_36.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_36.stopped')
                        # update status
                        text_36.status = FINISHED
                        text_36.setAutoDraw(False)
                
                # *text_16* updates
                
                # if text_16 is starting this frame...
                if text_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_16.frameNStart = frameN  # exact frame index
                    text_16.tStart = t  # local t and not account for scr refresh
                    text_16.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_16.started')
                    # update status
                    text_16.status = STARTED
                    text_16.setAutoDraw(True)
                
                # if text_16 is active this frame...
                if text_16.status == STARTED:
                    # update params
                    pass
                
                # if text_16 is stopping this frame...
                if text_16.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_16.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        text_16.tStop = t  # not accounting for scr refresh
                        text_16.tStopRefresh = tThisFlipGlobal  # on global time
                        text_16.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_16.stopped')
                        # update status
                        text_16.status = FINISHED
                        text_16.setAutoDraw(False)
                
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
                for thisComponent in communitynotesComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "communitynotes" ---
            for thisComponent in communitynotesComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('communitynotes.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from LSL_communtiyNotes
            outlet.push_sample(x=[marker7])  # Push event marker. Baseline==1
            
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
        # completed 1.0 repeats of 'trials_11'
        
        
        # --- Prepare to start Routine "Revision" ---
        continueRoutine = True
        # update component parameters for each repeat
        text_37.setText(Statement)
        thisExp.addData('Revision.started', globalClock.getTime(format='float'))
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
        block4.addData('slider_3.response', slider_3.getRating())
        block4.addData('slider_3.rt', slider_3.getRT())
        block4.addData('slider_4.response', slider_4.getRating())
        block4.addData('slider_4.rt', slider_4.getRT())
        block4.addData('slider_5.response', slider_5.getRating())
        block4.addData('slider_5.rt', slider_5.getRT())
        block4.addData('button.numClicks', button.numClicks)
        if button.numClicks:
           block4.addData('button.timesOn', button.timesOn)
           block4.addData('button.timesOff', button.timesOff)
        else:
           block4.addData('button.timesOn', "")
           block4.addData('button.timesOff', "")
        # the Routine "Revision" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_12 = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('feedback.xlsx', selection=useRows5),
            seed=None, name='trials_12')
        thisExp.addLoop(trials_12)  # add the loop to the experiment
        thisTrial_12 = trials_12.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_12.rgb)
        if thisTrial_12 != None:
            for paramName in thisTrial_12:
                globals()[paramName] = thisTrial_12[paramName]
        
        for thisTrial_12 in trials_12:
            currentLoop = trials_12
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_12.rgb)
            if thisTrial_12 != None:
                for paramName in thisTrial_12:
                    globals()[paramName] = thisTrial_12[paramName]
            
            # --- Prepare to start Routine "break_short" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('break_short.started', globalClock.getTime(format='float'))
            text_12.setText('')
            # keep track of which components have finished
            break_shortComponents = [text_12]
            for thisComponent in break_shortComponents:
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
            
            # --- Run Routine "break_short" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_12* updates
                
                # if text_12 is starting this frame...
                if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_12.frameNStart = frameN  # exact frame index
                    text_12.tStart = t  # local t and not account for scr refresh
                    text_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_12.started')
                    # update status
                    text_12.status = STARTED
                    text_12.setAutoDraw(True)
                
                # if text_12 is active this frame...
                if text_12.status == STARTED:
                    # update params
                    pass
                
                # if text_12 is stopping this frame...
                if text_12.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_12.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text_12.tStop = t  # not accounting for scr refresh
                        text_12.tStopRefresh = tThisFlipGlobal  # on global time
                        text_12.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_12.stopped')
                        # update status
                        text_12.status = FINISHED
                        text_12.setAutoDraw(False)
                
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
                for thisComponent in break_shortComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "break_short" ---
            for thisComponent in break_shortComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('break_short.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "Feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Feedback.started', globalClock.getTime(format='float'))
            text_23.setText(feedback)
            # keep track of which components have finished
            FeedbackComponents = [text_24, text_23]
            for thisComponent in FeedbackComponents:
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
            
            # --- Run Routine "Feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_24* updates
                
                # if text_24 is starting this frame...
                if text_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_24.frameNStart = frameN  # exact frame index
                    text_24.tStart = t  # local t and not account for scr refresh
                    text_24.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_24, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_24.started')
                    # update status
                    text_24.status = STARTED
                    text_24.setAutoDraw(True)
                
                # if text_24 is active this frame...
                if text_24.status == STARTED:
                    # update params
                    pass
                
                # if text_24 is stopping this frame...
                if text_24.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_24.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_24.tStop = t  # not accounting for scr refresh
                        text_24.tStopRefresh = tThisFlipGlobal  # on global time
                        text_24.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_24.stopped')
                        # update status
                        text_24.status = FINISHED
                        text_24.setAutoDraw(False)
                
                # *text_23* updates
                
                # if text_23 is starting this frame...
                if text_23.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_23.frameNStart = frameN  # exact frame index
                    text_23.tStart = t  # local t and not account for scr refresh
                    text_23.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_23, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_23.started')
                    # update status
                    text_23.status = STARTED
                    text_23.setAutoDraw(True)
                
                # if text_23 is active this frame...
                if text_23.status == STARTED:
                    # update params
                    pass
                
                # if text_23 is stopping this frame...
                if text_23.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_23.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_23.tStop = t  # not accounting for scr refresh
                        text_23.tStopRefresh = tThisFlipGlobal  # on global time
                        text_23.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_23.stopped')
                        # update status
                        text_23.status = FINISHED
                        text_23.setAutoDraw(False)
                
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
                for thisComponent in FeedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Feedback" ---
            for thisComponent in FeedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Feedback.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from Lsl_Feedback
            outlet.push_sample(x=[marker8])  # Push event marker. Baseline==1
            
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_12'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'block4'
    
    
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
