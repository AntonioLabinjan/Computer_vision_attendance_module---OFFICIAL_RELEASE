import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, send_file, request, redirect, url_for, jsonify, session, render_template_string, jsonify
import csv
import io
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
import datetime
from datetime import datetime, timedelta
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import re
from flask import flash
from flask_mail import Mail, Message
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import matplotlib.pyplot as plt
import seaborn as sns
from flask import send_file
import io
from collections import Counter 
import base64
import io
from dotenv import load_dotenv
import os
import sqlite3
from werkzeug.security import generate_password_hash
from bs4 import BeautifulSoup
import requests
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import faiss
