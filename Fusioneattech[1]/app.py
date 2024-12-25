import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, EmailField
from wtforms.validators import DataRequired, Email, Length
import random
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# SQLite Configuration (use PostgreSQL/MySQL for production)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fusionneat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = True  # Secure session cookies in production
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Zoho Mail Configuration
ZOHO_EMAIL = os.getenv("ZOHO_EMAIL")
ZOHO_PASSWORD = os.getenv("ZOHO_PASSWORD")
ZOHO_SMTP_SERVER = "smtppro.zoho.in"
ZOHO_SMTP_PORT = 465

# Load the saved chatbot model
model = load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as t_file:
    tokenizer = pickle.load(t_file)
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)
with open("dataset.json", "r") as file:
    data = json.load(file)
responses = {item["intent"]: item["response"] for item in data["data"]}

# Models
class ContactQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class NewsletterSubscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    subscribed_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)

# Create admin users with initial data
def create_admin():
    admins = [
        {'username': 'fusionneat_admin', 'password': '1Om2Hari3@Vuday', 'email': 'admin@example.com'},
        {'username': 'fusionneat_om', 'password': 'om@fusionneattech', 'email': 'ombaikerikar1@gmail.com'},
        {'username': 'fusionneat_hari', 'password': 'hari@fusionneattech', 'email': 'harimarathi224@gmail.com'},
        {'username': 'fusionneat_uday', 'password': 'uday@fusionneattech', 'email': 'vuday8370@gmail.com'}
    ]

    for admin_data in admins:
        admin = Admin.query.filter_by(username=admin_data['username']).first()
        if not admin:
            hashed_password = bcrypt.generate_password_hash(admin_data['password']).decode('utf-8')
            new_admin = Admin(
                username=admin_data['username'],
                email=admin_data['email'],
                password_hash=hashed_password
            )
            db.session.add(new_admin)

    db.session.commit()

# Forms
class AdminLoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=50)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=50)])
    submit = SubmitField('Login')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    message = StringField('Message', validators=[DataRequired(), Length(min=10, max=500)])
    submit = SubmitField('Send')

class NewsletterForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Subscribe')

# Email sending function
def send_email(subject, body, recipient):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = ZOHO_EMAIL
        msg['To'] = recipient

        with smtplib.SMTP_SSL(ZOHO_SMTP_SERVER, ZOHO_SMTP_PORT) as server:
            server.login(ZOHO_EMAIL, ZOHO_PASSWORD)
            server.sendmail(ZOHO_EMAIL, recipient, msg.as_string())
    except Exception as e:
        raise Exception(f"Failed to send email: {e}")

# Route for chatbot response
def get_response(query):
    sequence = tokenizer.texts_to_sequences([query])
    padded = pad_sequences(sequence, maxlen=model.input_shape[1], padding="post")
    prediction = model.predict(padded)
    predicted_intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return responses.get(predicted_intent, "Sorry, I couldn't understand your question.")

# Routes
@app.route('/')
def home():
    form = ContactForm()
    return render_template('index.html', form=form)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        message = form.message.data
        try:
            new_query = ContactQuery(name=name, email=email, message=message)
            db.session.add(new_query)
            db.session.commit()
            flash("Your query has been submitted successfully!", "success")
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('home'))
    return render_template('index.html', form=form)

@app.route('/subscribe', methods=['POST'])
def subscribe():
    form = NewsletterForm()
    if form.validate_on_submit():
        email = form.email.data
        try:
            new_subscriber = NewsletterSubscriber(email=email)
            db.session.add(new_subscriber)
            db.session.commit()
            flash("You have successfully subscribed to the newsletter!", "success")
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
    return redirect(url_for('home'))

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    form = AdminLoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        admin = Admin.query.filter_by(username=username).first()

        if admin and bcrypt.check_password_hash(admin.password_hash, password):
            otp = random.randint(100000, 999999)
            session['otp'] = otp
            session['admin_email'] = admin.email
            session['admin_id'] = admin.id

            try:
                send_email("Your Admin Dashboard OTP", f"Your OTP is {otp}.", admin.email)
                flash("An OTP has been sent to your registered email. Please verify.", "info")
                return redirect(url_for('verify_otp'))
            except Exception as e:
                flash(f"Error sending OTP: {str(e)}", "danger")
        else:
            flash("Invalid username or password.", "danger")
    return render_template('admin_login.html', form=form)

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form['otp']
        if 'otp' in session and int(entered_otp) == session['otp']:
            session.pop('otp')
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid OTP. Please try again.", "danger")
    return render_template('verify_otp.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        flash("Please log in to access the admin dashboard.", "warning")
        return redirect(url_for('admin_login'))

    queries = ContactQuery.query.order_by(ContactQuery.created_at.desc()).all()
    subscribers = NewsletterSubscriber.query.order_by(NewsletterSubscriber.subscribed_at.desc()).all()
    return render_template('admin_dashboard.html', queries=queries, subscribers=subscribers)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('admin_login'))

# Chatbot route
@app.route('/chat', methods=['GET'])
def chat():
    user_query = request.args.get("msg")
    if user_query:
        response = get_response(user_query)
        return jsonify({"response": response})
    return jsonify({"response": "Sorry, I couldn't understand your question."})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin()
    app.run(debug=True, ssl_context='adhoc')
