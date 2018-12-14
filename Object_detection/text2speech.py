from win32com.client import constants, Dispatch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-o', action='store', dest='o', type=str, required=True,
                    default='hello world')
inargs = parser.parse_args()
arg_str = inargs.o 


speaker = Dispatch("SAPI.SpVoice")
speaker.Speak(arg_str)
del speaker