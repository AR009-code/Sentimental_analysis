from django.shortcuts import render, redirect
from django.http import HttpResponse
from main import *
from . import utils
import numpy as np

max_emotion={
    'ANGRY':0,
    'DISGUST':0,
    'FEAR':0,
    'HAPPY':0,
    'SAD':0,
    'SURPRISE':0,
    'NEUTRAL':0,
}
emotions=['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL', 'DOMINANT']

# Create your views here.

def home(req):
    return render(req, 'html/layout.html',{'run_test_btn':True, 'ppt': False, 'got_result':False})

def run_test(req):
    table= camCapture()

    for row in table:
        row['emotion']['angry']= round(row['emotion']['angry'], 3)
        row['emotion']['disgust']= round(row['emotion']['disgust'], 3)
        row['emotion']['fear']= round(row['emotion']['fear'], 3)
        row['emotion']['happy']= round(row['emotion']['happy'], 3)
        row['emotion']['sad']= round(row['emotion']['sad'], 3)
        row['emotion']['surprise']= round(row['emotion']['surprise'], 3)
        row['emotion']['neutral']= round(row['emotion']['neutral'], 3)        

    for face in emotions:
        if(face != 'DOMINANT'):

            max_emotion[face]= max([x['emotion'][face.lower()] for x in table ])

    createCSV(table)
    
    x=[ max_emotion[x] for x in max_emotion.keys()]
    labels= [ x for x in max_emotion.keys()]

    print('x= ',x, 'labels= ', labels)
    chart= utils.get_plot(x, labels)
        
    return render(req, 'html/layout.html',{'run_test_btn':False, 'ppt': False, 'got_result':True, 'emotions':emotions,  'table': table, 'chart':chart })

def ppt(req):
    return render(req, 'html/layout.html',{'run_test_btn':False, 'ppt': True, 'got_result': False})


def result(req):
    return render(req)
