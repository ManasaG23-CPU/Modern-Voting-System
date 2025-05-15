import logging
import re
import time
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import json
import os
from PIL import Image
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
from werkzeug.utils import secure_filename
from iris_utils import capture_and_save_iris, compare_iris

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key in production

# Email configurations
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = 'daminmain@gmail.com'
EMAIL_PASSWORD = 'kpqtxqskedcykwjz'

# Image upload configurations
UPLOAD_FOLDER = 'static/candidates'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class BiometricAuthenticator:
    def __init__(self, db_path='user_biometrics.json', candidates_path='candidates.json', locations_path='vote_locations.json'):
        self.db_path = db_path
        self.candidates_path = candidates_path
        self.locations_path = locations_path
        self.ensure_db_exists()
        self.ensure_candidates_db_exists()
        self.ensure_locations_db_exists()
        self.fingerprint_model = self.load_fingerprint_model()
        self.iris_model = self.load_iris_model()
        self.twilio_client = Client('twilioAccount', 'twilioPassKey')
        self.twilio_number = 'twilioNumber'
        self.debug_dir = 'debug_images'
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

    def ensure_db_exists(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump({}, f)

    def ensure_candidates_db_exists(self):
        if not os.path.exists(self.candidates_path):
            with open(self.candidates_path, 'w') as f:
                json.dump([], f)

    def ensure_locations_db_exists(self):
        if not os.path.exists(self.locations_path):
            with open(self.locations_path, 'w') as f:
                json.dump([], f)

    def load_fingerprint_model(self):
        try:
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            return tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        except Exception as e:
            logger.error(f"Failed to load ResNet50 model for fingerprints: {e}")
            raise

    def load_iris_model(self):
        try:
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            return tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        except Exception as e:
            logger.error(f"Failed to load ResNet50 model for iris: {e}")
            raise

    def extract_iris_features(self, image_data):
        try:
            logger.debug("Starting iris feature extraction")
            image_data = base64.b64decode(image_data.split(',')[1])
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to decode iris image")
                return None
            debug_path = os.path.join(self.debug_dir, f'iris_{int(time.time())}.png')
            cv2.imwrite(debug_path, image)
            logger.debug(f"Saved iris debug image: {debug_path}")
            img_resized = cv2.resize(image, (224, 224))
            img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
            features = self.iris_model.predict(img_preprocessed)
            logger.debug(f"Iris features extracted successfully, length: {len(features.flatten())}")
            return features.flatten().tolist()
        except Exception as e:
            logger.error(f"Iris extraction failed: {e}")
            return None

    def extract_fingerprint_features(self, fingerprint_image):
        try:
            logger.debug("Starting fingerprint feature extraction")
            img = Image.open(fingerprint_image)
            img_array = np.array(img)
            debug_path = os.path.join(self.debug_dir, f'fingerprint_{int(time.time())}.png')
            cv2.imwrite(debug_path, img_array if len(img_array.shape) == 3 else cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR))
            logger.debug(f"Saved fingerprint debug image: {debug_path}")
            if len(img_array.shape) == 3:
                gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:

                
                gray_image = img_array
            rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
            img_resized = cv2.resize(rgb_image, (224, 224))
            img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
            features = self.fingerprint_model.predict(img_preprocessed)
            logger.debug(f"Fingerprint features extracted successfully, length: {len(features.flatten())}")
            return features.flatten().tolist()
        except Exception as e:
            logger.error(f"Fingerprint extraction failed: {e}")
            return None

    def fusion_features(self, iris_features, fingerprint_features):
        iris_norm = np.array(iris_features) / np.linalg.norm(iris_features)
        fingerprint_norm = np.array(fingerprint_features) / np.linalg.norm(fingerprint_features)
        fused = np.concatenate([0.6 * iris_norm, 0.4 * fingerprint_norm])
        logger.debug(f"Fused features length: {len(fused)}")
        return fused.tolist()

    def validate_aadhar(self, aadhar_number):
        if not re.match(r'^\d{12}$', aadhar_number):
            return False
        return True

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def register_user(self, username, aadhar_number, mobile_number, iris_image_data, fingerprint_image):
        if not self.validate_aadhar(aadhar_number):
            return False, "Invalid Aadhar number. It must be a 12-digit number."

        with open(self.db_path, 'r') as f:
            users = json.load(f)

        for user_data in users.values():
            if user_data['aadhar_number'] == aadhar_number:
                return False, "Aadhar number already registered. Please use a unique Aadhar number."

        iris_features = self.extract_iris_features(iris_image_data)
        fingerprint_features = self.extract_fingerprint_features(fingerprint_image)

        if not iris_features or not fingerprint_features:
            return False, "Feature extraction failed. Please ensure clear iris and fingerprint captures."

        fused_features = self.fusion_features(iris_features, fingerprint_features)

        if username in users:
            return False, "Username already exists. Please choose a different username."

        users[username] = {
            'aadhar_number': aadhar_number,
            'mobile_number': mobile_number,
            'fused_features': fused_features,
            'has_voted': False,
            'vote': None
        }

        with open(self.db_path, 'w') as f:
            json.dump(users, f)

        return True, "User registered successfully! Please proceed to OTP verification."

    def authenticate_user(self, aadhar_number, mobile_number, iris_image_data, fingerprint_image):
        iris_features = self.extract_iris_features(iris_image_data)
        fingerprint_features = self.extract_fingerprint_features(fingerprint_image)

        if not iris_features or not fingerprint_features:
            return None, "Feature extraction failed. Please ensure clear iris and fingerprint captures."

        input_features = self.fusion_features(iris_features, fingerprint_features)

        with open(self.db_path, 'r') as f:
            users = json.load(f)

        logger.debug(f"Attempting authentication for Aadhar: {aadhar_number}")
        user_found = False
        for username, user_data in users.items():
            if user_data['aadhar_number'] == aadhar_number and user_data['mobile_number'] == mobile_number:
                user_found = True
                similarity = self.compute_similarity(input_features, user_data['fused_features'])
                logger.debug(f"Similarity score for user {username} (Aadhar: {aadhar_number}): {similarity}")
                if similarity > 0.70:
                    session['authenticated_user'] = username
                    logger.info(f"Authentication successful for user: {username}")
                    return username, "Biometric Authentication Successful! Please proceed to voting."
                else:
                    logger.error(f"Authentication failed for user {username}: Similarity score {similarity} below threshold")
                    return None, "Authentication Failed: Biometric mismatch. Please recapture using the same iris and finger as during registration."
        
        if not user_found:
            logger.error(f"No user found with Aadhar: {aadhar_number} and mobile: {mobile_number}")
            return None, "Authentication Failed: Invalid Aadhar number or mobile number."
        return None, "Authentication Failed: Biometric mismatch."

    def compute_similarity(self, features1, features2):
        features1 = np.array(features1)
        features2 = np.array(features2)
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def send_otp(self, mobile_number, email):
        otp = str(np.random.randint(100000, 999999))
        session['otp'] = otp
        session['otp_timestamp'] = time.time()

        formatted_number = mobile_number.replace(" ", "").strip()
        if not formatted_number.startswith("+91"):
            formatted_number = "+91" + formatted_number.lstrip("+")
        if not re.match(r'^\+91\d{10}$', formatted_number):
            logger.error(f"Invalid mobile number format: {formatted_number}")
            raise ValueError("Mobile number must be a 10-digit number after +91")

        logger.debug(f"Formatted mobile number for SMS: {formatted_number}")
        sms_success = False
        try:
            message = self.twilio_client.messages.create(
                body=f"Your OTP for voting system is {otp}",
                from_=self.twilio_number,
                to=formatted_number
            )
            logger.info(f"SMS OTP sent successfully to {formatted_number}")
            sms_success = True
        except Exception as e:
            logger.error(f"Failed to send SMS OTP to {formatted_number}: {str(e)}")

        try:
            self.send_email(email, "Voting System OTP", f"Your OTP is {otp}")
            logger.info(f"Email OTP sent successfully to {email}")
            if not sms_success:
                logger.warning("SMS OTP failed, but email OTP sent as fallback")
        except Exception as e:
            logger.error(f"Failed to send email OTP to {email}: {str(e)}")
            if not sms_success:
                raise Exception("Both SMS and email OTP sending failed")

        return otp

    def send_email(self, to_email, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        logger.debug(f"Sending email to: {to_email}")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())

    def verify_otp(self, input_otp):
        if 'otp' not in session or 'otp_timestamp' not in session:
            return False, "OTP not generated or expired"
        
        stored_otp = session['otp']
        timestamp = session['otp_timestamp']
        
        if time.time() - timestamp > 300:
            session.pop('otp', None)
            session.pop('otp_timestamp', None)
            return False, "OTP expired"
        
        if input_otp == stored_otp:
            session.pop('otp', None)
            session.pop('otp_timestamp', None)
            return True, "OTP verified successfully"
        
        return False, "Invalid OTP"

    def cast_vote(self, username, candidate_name, latitude=None, longitude=None):
        try:
            with open(self.db_path, 'r') as f:
                users = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read user_biometrics.json: {e}")
            return False, "Failed to access user data"
        
        if username not in users:
            return False, "User not found"
        
        if users[username]['has_voted']:
            return False, "User has already voted"

        try:
            with open(self.candidates_path, 'r') as f:
                candidates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read candidates.json: {e}")
            return False, "Failed to access candidate data"
        
        if not any(c['name'] == candidate_name for c in candidates):
            return False, "Invalid candidate"

        users[username]['has_voted'] = True
        users[username]['vote'] = candidate_name
        try:
            with open(self.db_path, 'w') as f:
                json.dump(users, f)
        except Exception as e:
            logger.error(f"Failed to write user_biometrics.json: {e}")
            return False, "Failed to save vote"

        try:
            with open(self.locations_path, 'r') as f:
                locations = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read vote_locations.json: {e}")
            locations = []

        location_entry = {
            'username': username,
            'candidate': candidate_name,
            'latitude': latitude if latitude else None,
            'longitude': longitude if longitude else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        }
        locations.append(location_entry)
        try:
            with open(self.locations_path, 'w') as f:
                json.dump(locations, f)
        except Exception as e:
            logger.error(f"Failed to write vote_locations.json: {e}")
            logger.warning("Vote recorded, but location data not saved")

        return True, f"Vote for {candidate_name} cast successfully!"

    def get_voting_results(self):
        results = {'total_voters': 0, 'voted': 0, 'not_voted': 0, 'candidates': {}}
        try:
            with open(self.db_path, 'r') as f:
                users = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read user_biometrics.json: {e}")
            return results
        
        try:
            with open(self.candidates_path, 'r') as f:
                candidates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read candidates.json: {e}")
            return results
        
        for candidate in candidates:
            results['candidates'][candidate['name']] = 0

        results['total_voters'] = len(users)
        for user in users.values():
            if user.get('has_voted'):
                results['voted'] += 1
                if user.get('vote'):
                    results['candidates'][user['vote']] = results['candidates'].get(user['vote'], 0) + 1
            else:
                results['not_voted'] += 1
        
        return results

    def add_candidate(self, name, party, avatar_file, symbol_file, manifesto, old_name=None):
        try:
            with open(self.candidates_path, 'r') as f:
                candidates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read candidates.json: {e}")
            candidates = []
        
        if old_name != name and any(c['name'] == name for c in candidates):
            return False, "Candidate name already exists"

        new_candidate = {
            'name': name,
            'party': party,
            'manifesto': manifesto
        }

        if avatar_file and avatar_file.filename and self.allowed_file(avatar_file.filename):
            avatar_filename = secure_filename(f"{name}_avatar_{int(time.time())}.{avatar_file.filename.rsplit('.', 1)[1].lower()}")
            avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], avatar_filename)
            avatar_file.save(avatar_path)
            logger.debug(f"Saved avatar image: {avatar_path}")
            new_candidate['avatar'] = avatar_filename
        elif old_name:
            old_candidate = next((c for c in candidates if c['name'] == old_name), None)
            if old_candidate:
                new_candidate['avatar'] = old_candidate['avatar']
        else:
            return False, "Avatar image is required"

        if symbol_file and symbol_file.filename and self.allowed_file(symbol_file.filename):
            symbol_filename = secure_filename(f"{name}_symbol_{int(time.time())}.{symbol_file.filename.rsplit('.', 1)[1].lower()}")
            symbol_path = os.path.join(app.config['UPLOAD_FOLDER'], symbol_filename)
            symbol_file.save(symbol_path)
            logger.debug(f"Saved symbol image: {symbol_path}")
            new_candidate['symbol'] = symbol_filename
        elif old_name:
            old_candidate = next((c for c in candidates if c['name'] == old_name), None)
            if old_candidate:
                new_candidate['symbol'] = old_candidate['symbol']
        else:
            return False, "Symbol image is required"

        if old_name:
            old_candidate = next((c for c in candidates if c['name'] == old_name), None)
            if old_candidate:
                candidates = [c for c in candidates if c['name'] != old_name]
                if old_candidate['avatar'] != new_candidate['avatar']:
                    old_avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], old_candidate['avatar'])
                    if os.path.exists(old_avatar_path):
                        os.remove(old_avatar_path)
                        logger.debug(f"Deleted old avatar: {old_avatar_path}")
                if old_candidate['symbol'] != new_candidate['symbol']:
                    old_symbol_path = os.path.join(app.config['UPLOAD_FOLDER'], old_candidate['symbol'])
                    if os.path.exists(old_symbol_path):
                        os.remove(old_symbol_path)
                        logger.debug(f"Deleted old symbol: {old_symbol_path}")

        candidates.append(new_candidate)
        try:
            with open(self.candidates_path, 'w') as f:
                json.dump(candidates, f)
        except Exception as e:
            logger.error(f"Failed to write candidates.json: {e}")
            return False, "Failed to save candidate data"
        
        return True, "Candidate saved successfully"

    def delete_candidate(self, name):
        try:
            with open(self.candidates_path, 'r') as f:
                candidates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read candidates.json: {e}")
            return False, "Failed to access candidate data"
        
        candidate = next((c for c in candidates if c['name'] == name), None)
        if not candidate:
            return False, "Candidate not found"
        
        candidates = [c for c in candidates if c['name'] != name]
        for img_type in ['avatar', 'symbol']:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate[img_type])
            if os.path.exists(img_path):
                os.remove(img_path)
                logger.debug(f"Deleted {img_type} image: {img_path}")
        
        try:
            with open(self.candidates_path, 'w') as f:
                json.dump(candidates, f)
        except Exception as e:
            logger.error(f"Failed to write candidates.json: {e}")
            return False, "Failed to save candidate data"
        
        return True, "Candidate deleted successfully"

    def get_candidates(self):
        try:
            with open(self.candidates_path, 'r') as f:
                candidates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read candidates.json: {e}")
            return []
        
        for candidate in candidates:
            candidate['avatar_path'] = f"/{app.config['UPLOAD_FOLDER']}/{candidate['avatar']}"
            candidate['symbol_path'] = f"/{app.config['UPLOAD_FOLDER']}/{candidate['symbol']}"
        return candidates

    def get_vote_locations(self):
        try:
            with open(self.locations_path, 'r') as f:
                locations = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read vote_locations.json: {e}")
            return []
        return locations

authenticator = BiometricAuthenticator()

@app.route('/')
def landing_page():
    return render_template('landing.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Log all received form and file data
            logger.debug(f"Form data received: {dict(request.form)}")
            logger.debug(f"Files received: {dict(request.files)}")

            # Safely access form fields
            username = request.form.get('username')
            aadhar_number = request.form.get('aadhar_number')
            mobile_number = request.form.get('mobile_number')
            iris_image_data = request.form.get('iris_image_data')
            fingerprint_image = request.files.get('fingerprint_image')

            # Validate required fields
            required_fields = {
                'username': username,
                'aadhar_number': aadhar_number,
                'mobile_number': mobile_number,
                'iris_image_data': iris_image_data,
                'fingerprint_image': fingerprint_image
            }
            missing_fields = [key for key, value in required_fields.items() if not value or (key == 'fingerprint_image' and not value.filename)]
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                flash(error_msg, "error")
                return render_template('register.html', error=error_msg)

            # Validate fingerprint image format
            if not authenticator.allowed_file(fingerprint_image.filename):
                error_msg = "Invalid fingerprint image format. Please upload PNG, JPG, or JPEG."
                logger.error(error_msg)
                flash(error_msg, "error")
                return render_template('register.html', error=error_msg)

            # Validate iris_image_data (basic check for base64 format)
            if not iris_image_data.startswith('data:image/'):
                error_msg = "Invalid iris image data. Please capture a valid face image."
                logger.error(error_msg)
                flash(error_msg, "error")
                return render_template('register.html', error=error_msg)

            logger.debug(f"Registering user: {username}, Aadhar: {aadhar_number}")
            success, message = authenticator.register_user(
                username, aadhar_number, mobile_number, iris_image_data, fingerprint_image
            )
            if success:
                session['username'] = username
                logger.debug(f"Session after registration: {session}")
                try:
                    authenticator.send_otp(mobile_number, "your-gmail@.com")
                    logger.info(f"User {username} registered successfully, OTP sent")
                    return redirect(url_for('verify_otp', action='register'))
                except Exception as e:
                    logger.error(f"Registration failed due to OTP sending error: {str(e)}")
                    flash("Registration successful, but OTP sending failed. Please try again.", "error")
                    return render_template('register.html', error="OTP sending failed")
            logger.error(f"Registration failed: {message}")
            flash(message, "error")
            return render_template('register.html', error=message)
        except Exception as e:
            logger.error(f"Error in register route: {str(e)}")
            flash("An error occurred during registration. Please try again.", "error")
            return render_template('register.html', error="An error occurred")
    
    return render_template('register.html')


@app.route("/register-iris", methods=["POST"])
def register_iris():
    if request.method == "POST":
        try:
            username = request.form.get("username")
            if not username:
                return jsonify({
                    "status": "error",
                    "message": "Username is required"
                }), 400

            success = capture_and_save_iris(username)
            if success:
                return jsonify({
                    "status": "success",
                    "message": "✅ Capture!"
                }), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": "❌ Failed to Capture"
                }), 500
        except KeyError:
            return jsonify({
                "status": "error",
                "message": "Username field is missing"
            }), 400
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            logger.debug(f"Form data received: {dict(request.form)}")
            phone_number = request.form.get('phone_number')
            if not phone_number:
                error_msg = "Missing phone number."
                logger.error(error_msg)
                flash(error_msg, "error")
                return render_template('login.html', error=error_msg)

            session['login_data'] = {'phone_number': phone_number}
            logger.debug(f"Stored login_data in session: {session['login_data']}")

            try:
                authenticator.send_otp(phone_number, "your-gmail.com")
                logger.info(f"OTP sent successfully to {phone_number}")
                return redirect(url_for('verify_otp', action='login'))
            except Exception as e:
                logger.error(f"Failed to send OTP: {str(e)}")
                flash("Failed to send OTP. Please try again.", "error")
                return render_template('login.html', error="Failed to send OTP")
        except Exception as e:
            logger.error(f"Error in login route: {str(e)}")
            flash("An error occurred during login. Please try again.", "error")
            return render_template('login.html', error="An error occurred")
    
    return render_template('login.html')

@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        try:
            logger.debug(f"Form data received: {dict(request.form)}")
            logger.debug(f"Files received: {dict(request.files)}")
            logger.debug(f"Session data: {dict(session)}")

            aadhar_number = request.form.get('aadhar_number')
            iris_image_data = request.form.get('iris_image_data')
            fingerprint_image = request.files.get('fingerprint_image')

            required_fields = {
                'aadhar_number': aadhar_number,
                'iris_image_data': iris_image_data,
                'fingerprint_image': fingerprint_image
            }
            missing_fields = [key for key, value in required_fields.items() if not value or (key == 'fingerprint_image' and not value.filename)]
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                flash(error_msg, "error")
                return render_template('authenticate.html', error=error_msg)

            if not authenticator.allowed_file(fingerprint_image.filename):
                error_msg = "Invalid fingerprint image format. Please upload PNG, JPG, or JPEG."
                logger.error(error_msg)
                flash(error_msg, "error")
                return render_template('authenticate.html', error=error_msg)

            filename = secure_filename(fingerprint_image.filename)
            temp_path = os.path.join('temp', filename)
            os.makedirs('temp', exist_ok=True)
            fingerprint_image.save(temp_path)
            logger.debug(f"Saved fingerprint image to: {temp_path}")

            login_data = session.get('login_data', {})
            phone_number = login_data.get('phone_number')
            if not phone_number:
                error_msg = "Phone number not found in session. Please log in again."
                logger.error(error_msg)
                flash(error_msg, "error")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return redirect(url_for('login'))

            session['login_data'] = {
                'phone_number': phone_number,
                'aadhar_number': aadhar_number,
                'iris_image_data': iris_image_data,
                'fingerprint_path': temp_path
            }
            logger.debug(f"Updated login_data in session: {session['login_data']}")

            try:
                result, message = authenticator.authenticate_user(
                    aadhar_number,
                    phone_number,
                    iris_image_data,
                    open(temp_path, 'rb')
                )
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                flash("Authentication failed due to an error.", "error")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                session.pop('login_data', None)
                return render_template('authenticate.html', error="Authentication failed")

            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Deleted temporary fingerprint image: {temp_path}")

            if result:
                session['username'] = result
                session.pop('login_data', None)
                logger.info(f"User {result} authenticated successfully")
                return redirect(url_for('vote'))
            logger.error(f"Authentication failed: {message}")
            flash(message, "error")
            session.pop('login_data', None)
            return render_template('result.html', result=message)
        except Exception as e:
            logger.error(f"Error in authenticate route: {str(e)}")
            flash("An error occurred during authentication. Please try again.", "error")
            if 'login_data' in session and 'fingerprint_path' in session['login_data']:
                temp_path = session['login_data'].get('fingerprint_path')
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            session.pop('login_data', None)
            return render_template('authenticate.html', error="An error occurred")
    
    return render_template('authenticate.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    action = request.args.get('action')
    if request.method == 'POST':
        try:
            logger.debug(f"Form data received: {dict(request.form)}")
            logger.debug(f"Session data: {dict(session)}")

            otp = request.form.get('otp')
            if not otp:
                error_msg = "Please enter the OTP."
                logger.error(error_msg)
                flash(error_msg, "error")
                template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
                return render_template(template, action=action, error=error_msg)

            stored_otp = session.get('otp')
            otp_timestamp = session.get('otp_timestamp')
            current_time = time.time()

            if not stored_otp or not otp_timestamp:
                error_msg = "No OTP found. Please request a new OTP."
                logger.error(error_msg)
                flash(error_msg, "error")
                template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
                return render_template(template, action=action, error=error_msg)

            if current_time - otp_timestamp > 300:
                error_msg = "OTP has expired. Please request a new OTP."
                logger.error(error_msg)
                flash(error_msg, "error")
                session.pop('otp', None)
                session.pop('otp_timestamp', None)
                template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
                return render_template(template, action=action, error=error_msg)

            if otp != stored_otp:
                error_msg = "Invalid OTP. Please try again."
                logger.error(error_msg)
                flash(error_msg, "error")
                template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
                return render_template(template, action=action, error=error_msg)

            session.pop('otp', None)
            session.pop('otp_timestamp', None)

            if action == 'login':
                if not session.get('login_data', {}).get('phone_number'):
                    error_msg = "Phone number not found. Please log in again."
                    logger.error(error_msg)
                    flash(error_msg, "error")
                    return redirect(url_for('login'))
                logger.info("OTP verified for login, redirecting to authenticate")
                return redirect(url_for('authenticate'))
            elif action == 'register':
                if not session.get('username'):
                    error_msg = "User data not found. Please register again."
                    logger.error(error_msg)
                    flash(error_msg, "error")
                    return redirect(url_for('register'))
                logger.info("OTP verified for registration, redirecting to registration_success")
                return redirect(url_for('registration_success'))
            else:
                error_msg = "Invalid action."
                logger.error(error_msg)
                flash(error_msg, "error")
                template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
                return render_template(template, action=action, error=error_msg)
        except Exception as e:
            logger.error(f"Error in verify_otp route: {str(e)}")
            flash("An error occurred during OTP verification. Please try again.", "error")
            template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
            return render_template(template, action=action, error="An error occurred")
    
    template = 'verify_otp.html' if action == 'login' else 'verify_otp.html'
    return render_template(template, action=action)

@app.route('/registration_success')
def registration_success():
    if 'username' not in session:
        logger.error("Unauthorized access to /registration_success")
        flash("Please register or log in to view this page.", "error")
        return redirect(url_for('login'))
    
    username = session['username']
    logger.info(f"Displaying registration success for user: {username}")
    return render_template('registration_success.html', username=username)

@app.route('/vote', methods=['GET', 'POST'])
def vote():
    if 'username' not in session:
        logger.error("Unauthorized access to /vote")
        flash("Please authenticate to vote.", "error")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        candidate_name = request.form.get('candidate')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        username = session['username']
        success, message = authenticator.cast_vote(username, candidate_name, latitude, longitude)
        if success:
            logger.info(f"Vote cast by {username} for {candidate_name}")
            flash(message, "success")
            return redirect(url_for('thankyou'))
        else:
            logger.error(f"Vote casting failed: {message}")
            flash(message, "error")
        return render_template('result.html', result=message)
    
    candidates = authenticator.get_candidates()
    return render_template('vote.html', candidates=candidates)

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['admin'] = True
            logger.info("Admin logged in successfully")
            return redirect(url_for('admin_dashboard'))
        logger.error("Admin login failed: Invalid credentials")
        flash("Invalid credentials", "error")
        return render_template('admin_login.html', error="Invalid credentials")
    
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin' not in session:
        logger.error("Unauthorized access to /admin_dashboard")
        flash("Please log in as admin.", "error")
        return redirect(url_for('admin_login'))
    
    results = authenticator.get_voting_results()
    candidates = authenticator.get_candidates()
    vote_locations = authenticator.get_vote_locations()
    return render_template('admin_dashboard.html', results=results, candidates=candidates, vote_locations=vote_locations)

@app.route('/add_candidate', methods=['GET', 'POST'])
def add_candidate():
    if 'admin' not in session:
        logger.error("Unauthorized access to /add_candidate")
        flash("Please log in as admin.", "error")
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        party = request.form.get('party')
        avatar_file = request.files.get('avatar')
        symbol_file = request.files.get('symbol')
        manifesto = request.form.get('manifesto')
        success, message = authenticator.add_candidate(name, party, avatar_file, symbol_file, manifesto)
        if success:
            logger.info(f"Candidate {name} added successfully")
            flash("Candidate added successfully", "success")
            return redirect(url_for('admin_dashboard'))
        logger.error(f"Failed to add candidate: {message}")
        flash(message, "error")
        return render_template('add_candidate.html', error=message)
    
    return render_template('add_candidate.html')

@app.route('/edit_candidate/<name>', methods=['GET', 'POST'])
def edit_candidate(name):
    if 'admin' not in session:
        logger.error("Unauthorized access to /edit_candidate")
        flash("Please log in as admin.", "error")
        return redirect(url_for('admin_login'))
    
    candidates = authenticator.get_candidates()
    candidate = next((c for c in candidates if c['name'] == name), None)
    if not candidate:
        logger.warning(f"Attempted to edit non-existent candidate: {name}")
        flash("Candidate not found", "error")
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        new_name = request.form.get('name')
        party = request.form.get('party')
        avatar_file = request.files.get('avatar')
        symbol_file = request.files.get('symbol')
        manifesto = request.form.get('manifesto')
        success, message = authenticator.add_candidate(new_name, party, avatar_file, symbol_file, manifesto, old_name=name)
        if success:
            logger.info(f"Candidate {new_name} updated successfully")
            flash("Candidate updated successfully", "success")
            return redirect(url_for('admin_dashboard'))
        logger.error(f"Failed to update candidate: {message}")
        flash(message, "error")
        return render_template('edit_candidate.html', candidate=candidate, error=message)
    
    return render_template('edit_candidate.html', candidate=candidate)

@app.route('/delete_candidate/<name>', methods=['POST'])
def delete_candidate(name):
    if 'admin' not in session:
        logger.error("Unauthorized access to /delete_candidate")
        flash("Please log in as admin.", "error")
        return redirect(url_for('admin_login'))
    
    success, message = authenticator.delete_candidate(name)
    if success:
        logger.info(f"Candidate {name} deleted successfully")
        flash("Candidate deleted successfully", "success")
    else:
        logger.error(f"Failed to delete candidate: {message}")
        flash(message, "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/view_candidate/<name>')
def view_candidate(name):
    if 'admin' not in session:
        logger.error("Unauthorized access to /view_candidate")
        flash("Please log in as admin.", "error")
        return redirect(url_for('admin_login'))
    
    candidates = authenticator.get_candidates()
    candidate = next((c for c in candidates if c['name'] == name), None)
    if not candidate:
        logger.warning(f"Attempted to view non-existent candidate: {name}")
        flash("Candidate not found", "error")
        return redirect(url_for('admin_dashboard'))
    
    return render_template('view_candidate.html', candidate=candidate)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        logger.error("Unauthorized access to /dashboard")
        flash("Please authenticate to view dashboard.", "error")
        return redirect(url_for('login'))
    
    username = session['username']
    try:
        with open('user_biometrics.json', 'r') as f:
            users = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read user_biometrics.json: {e}")
        flash("Failed to load user data.", "error")
        return redirect(url_for('login'))
    
    user_data = users.get(username, {})
    return render_template('landing.html', user=user_data)

@app.route('/thankyou', methods = ['GET', 'POST'])
def thankyou():
    return render_template('thank_you.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'username' not in session:
        logger.error("Unauthorized access to /feedback")
        flash("Please authenticate and vote to provide feedback.", "error")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        username = session['username']
        if not feedback_text:
            logger.error("Feedback is empty")
            flash("Please provide feedback.", "error")
            return render_template('feedback.html')
        
        success, message = authenticator.save_feedback(username, feedback_text)
        if success:
            logger.info(f"Feedback submitted by {username}")
            flash(message, "success")
            return redirect(url_for('landing_page'))
        else:
            logger.error(f"Feedback submission failed: {message}")
            flash(message, "error")
            return render_template('feedback.html')
    
    return render_template('feedback.html')
if __name__ == '__main__':
    app.run(debug=True)