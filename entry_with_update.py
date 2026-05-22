"""
Fooocus Design Tool — Entry Point
"""
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

print('Fooocus Design Tool — Starting...')
from launch import *
