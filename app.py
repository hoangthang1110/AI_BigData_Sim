import os
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use 'Agg' for non-interactive backends (for generating image files)
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime
# import sqlite3 # NO LONGER NEEDED FOR SQLITE
from PIL import Image

# Import Flask related modules
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash # For password hashing

# Import Flask-Login
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Import Flask-SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

# Import Flask-WTF and WTForms for forms
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, ValidationError
from wtforms.validators import DataRequired, EqualTo, Length # REMOVED Email validator

# Suppress warnings from tensorflow/keras
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import TensorFlow and Keras for image analysis
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results' # Folder to save chart images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 # 16 MB max per file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH # Set max upload size
app.config['SECRET_KEY'] = 'your_super_secret_key_here' # IMPORTANT: Change this to a strong, random key in production!

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Database Configuration (SQLAlchemy for PostgreSQL) ---
# Format: 'postgresql://user:password@host:port/database_name'
# Dùng aiuser và ai_password đã được cấp đủ quyền
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Thangpassword@localhost:5432/ai_image_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress warning
db = SQLAlchemy(app)

# --- Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to login page if not logged in
login_manager.login_message_category = 'info' # Flash message category

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    # email = db.Column(db.String(120), unique=True, nullable=False) # REMOVED EMAIL COLUMN
    password_hash = db.Column(db.String(256), nullable=False)

    __tablename__ = 'users'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Image Metadata Model ---
class ImageMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500), unique=True, nullable=False)
    category = db.Column(db.String(100)) # e.g., 'N/A' or user-defined
    uploaded_by_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    upload_timestamp = db.Column(db.String(50), nullable=False) # Store as string ISO format
    ai_predicted_category = db.Column(db.String(255))
    ai_prediction_confidence = db.Column(db.Float)
    ai_top_3_predictions = db.Column(db.Text) # Use Text for longer strings

    user = db.relationship('User', backref='images_uploaded')

    __tablename__ = 'image_metadata' # Explicit table name

    def __repr__(self):
        return f'<ImageMetadata {self.file_name} by User {self.uploaded_by_user_id}>'

# --- Forms (Flask-WTF) ---
class RegistrationForm(FlaskForm):
    username = StringField('Tên đăng nhập', validators=[DataRequired(), Length(min=2, max=64)])
    # email = StringField('Email', validators=[DataRequired(), Email()]) # REMOVED EMAIL FIELD
    password = PasswordField('Mật khẩu', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Xác nhận mật khẩu', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Đăng ký')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.')

    # def validate_email(self, email): # REMOVED VALIDATE_EMAIL METHOD
    #     user = User.query.filter_by(email=email.data).first()
    #     if user:
    #         raise ValidationError('Email đã tồn tại. Vui lòng sử dụng email khác.')

class LoginForm(FlaskForm):
    username = StringField('Tên đăng nhập', validators=[DataRequired()])
    password = PasswordField('Mật khẩu', validators=[DataRequired()])
    submit = SubmitField('Đăng nhập')

# --- Image Analysis Model ---
# Load ResNet50 model
model = ResNet50(weights='imagenet')

# --- Initialize the databases on app startup ---
with app.app_context():
    # Attempt to create tables. This is safe to run multiple times.
    # If the tables (users, image_metadata) already exist, SQLAlchemy won't recreate them.
    # If the 'users' table existed with an 'email' column, and you remove it,
    # you might need to manually drop and recreate the 'users' table
    # or use Alembic for database migrations. For simplicity, if you get an error,
    # consider dropping the 'users' table in pgAdmin and letting create_all() recreate it.
    db.create_all()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(image_path):
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = preprocess_input(x)
        x = tf.expand_dims(x, axis=0) # Add batch dimension

        predictions = model.predict(x, verbose=0) # Set verbose to 0 to suppress output
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Get top 3 predictions

        main_prediction = decoded_predictions[0]
        predicted_category = main_prediction[1]
        confidence = float(main_prediction[2])

        top_3_predictions_str = ", ".join([f"{name} ({score:.2f})" for (_, name, score) in decoded_predictions])

        return predicted_category, confidence, top_3_predictions_str
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return "Unknown", 0.0, "N/A"

def generate_charts(df):
    print(f"DEBUG: generate_charts called with DataFrame of shape: {df.shape}") # Add this line
    chart_paths = {}

    # Ensure RESULTS_FOLDER exists
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    if not df.empty:
        # 1. Bar chart for AI Predicted Category counts
        plt.figure(figsize=(10, 6))
        sns.countplot(y='ai_predicted_category', data=df, order=df['ai_predicted_category'].value_counts().index, palette='viridis')
        plt.title('Số lượng Hình ảnh theo Danh mục AI dự đoán')
        plt.xlabel('Số lượng')
        plt.ylabel('Danh mục AI dự đoán')
        plt.tight_layout()
        bar_chart_filename = 'bar_chart.png'
        bar_chart_path_full = os.path.join(RESULTS_FOLDER, bar_chart_filename)
        plt.savefig(bar_chart_path_full)
        plt.close()
        chart_paths['bar_chart'] = url_for('results_charts', filename=bar_chart_filename)

        # 2. Pie chart for AI Predicted Category percentages
        plt.figure(figsize=(8, 8))
        category_counts = df['ai_predicted_category'].value_counts()
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Phần trăm Hình ảnh theo Danh mục AI dự đoán')
        plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        pie_chart_filename = 'pie_chart.png'
        pie_chart_path_full = os.path.join(RESULTS_FOLDER, pie_chart_filename)
        plt.savefig(pie_chart_path_full)
        plt.close()
        chart_paths['pie_chart'] = url_for('results_charts', filename=pie_chart_filename)

        # 3. User uploads chart (Top 10 users)
        plt.figure(figsize=(10, 6))
        # Fetch usernames for display instead of just user IDs
        user_id_to_username = {u.id: u.username for u in User.query.all()}
        
        # Replace user IDs with usernames in the DataFrame for plotting
        df['uploaded_by_username'] = df['uploaded_by_user_id'].map(user_id_to_username).fillna('Unknown User')
        
        user_counts = df['uploaded_by_username'].value_counts().head(10)
        sns.barplot(x=user_counts.values, y=user_counts.index, palette='magma')
        plt.title('Top 10 Người dùng tải ảnh nhiều nhất')
        plt.xlabel('Số lượng ảnh')
        plt.ylabel('Tên Người dùng')
        plt.tight_layout()
        user_uploads_chart_filename = 'user_uploads_chart.png'
        user_uploads_chart_path_full = os.path.join(RESULTS_FOLDER, user_uploads_chart_filename)
        plt.savefig(user_uploads_chart_full)
        plt.close()
        chart_paths['user_uploads_chart'] = url_for('results_charts', filename=user_uploads_chart_filename)
    else:
        # If DataFrame is empty, ensure no old charts are displayed
        # Delete existing chart files if no data
        for f in os.listdir(RESULTS_FOLDER):
            if f.endswith('.png'):
                os.remove(os.path.join(RESULTS_FOLDER, f))
        print("DataFrame is empty, deleted old chart files.")

    return chart_paths

# --- Routes ---

@app.route('/')
@login_required # Only accessible if logged in
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # Removed email from User creation
        user = User(username=form.username.data)
        user.set_password(form.password.data)
        try:
            db.session.add(user)
            db.session.commit()
            flash('Tài khoản của bạn đã được tạo thành công! Bạn có thể đăng nhập ngay bâyi giờ.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback() # Rollback in case of error
            flash(f'Đã xảy ra lỗi khi đăng ký: {e}', 'danger')
            print(f"Registration error: {e}")
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Đăng nhập thành công!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Đăng nhập không thành công. Vui lòng kiểm tra tên đăng nhập và mật khẩu.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Bạn đã đăng xuất.', 'info')
    return redirect(url_for('login'))

@app.route('/upload_file', methods=['POST'])
@login_required # Requires login to upload images
def upload_file():
    if 'file[]' not in request.files:
        flash('Không có phần tệp nào', 'danger')
        return redirect(request.url)
    
    files = request.files.getlist('file[]')
    if not files or all(f.filename == '' for f in files):
        flash('Không có tệp nào được chọn hoặc tên tệp trống', 'danger')
        return redirect(request.url)

    uploaded_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                # Ensure unique filename to prevent overwriting
                base, ext = os.path.splitext(filename)
                unique_filename = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}{ext}" # Added microseconds for more uniqueness
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                # Analyze image
                predicted_category, confidence, top_3_predictions = analyze_image(filepath)
                
                # Create ImageMetadata object and add to database (using SQLAlchemy)
                new_image_metadata = ImageMetadata(
                    file_name=unique_filename,
                    image_path=filepath,
                    category='N/A', # Or allow user to set this later
                    uploaded_by_user_id=current_user.id,
                    upload_timestamp=datetime.now().isoformat(),
                    ai_predicted_category=predicted_category,
                    ai_prediction_confidence=confidence,
                    ai_top_3_predictions=top_3_predictions
                )
                db.session.add(new_image_metadata)
                db.session.commit() # Commit each image separately or commit all at once after loop

                uploaded_count += 1

            except Exception as e:
                db.session.rollback() # Rollback if an error occurs during commit
                flash(f'Lỗi khi xử lý tệp {file.filename}: {e}', 'warning')
                print(f"Error processing file {file.filename}: {e}")
        else:
            flash(f'Tệp {file.filename} không hợp lệ. Chỉ cho phép các định dạng {", ".join(ALLOWED_EXTENSIONS)}.', 'warning')
    
    if uploaded_count > 0:
        flash(f'Đã tải lên thành công {uploaded_count} ảnh và phân tích.', 'success')
    else:
        flash('Không có ảnh nào được tải lên hoặc phân tích thành công.', 'info')

    return redirect(url_for('results'))

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    # This route serves files from the UPLOAD_FOLDER
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results_charts/<filename>')
def results_charts(filename):
    # This route is specifically for chart images generated by the app, stored in RESULTS_FOLDER
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/results')
@login_required # Requires login to view results
def results():
    df_results = pd.DataFrame() # Initialize an empty DataFrame
    charts = {} # Initialize charts as an empty dictionary

    try:
        # Fetch all image metadata using SQLAlchemy
        all_images_metadata = ImageMetadata.query.all()
        
        if all_images_metadata:
            # Convert list of ImageMetadata objects to a list of dictionaries
            data_for_df = [
                {
                    'id': img.id,
                    'file_name': img.file_name,
                    'image_path': img.image_path,
                    'category': img.category,
                    'uploaded_by_user_id': img.uploaded_by_user_id,
                    'upload_timestamp': img.upload_timestamp,
                    'ai_predicted_category': img.ai_predicted_category,
                    'ai_prediction_confidence': img.ai_prediction_confidence,
                    'ai_top_3_predictions': img.ai_top_3_predictions
                }
                for img in all_images_metadata
            ]
            df_results = pd.DataFrame(data_for_df)
            print(f"DEBUG: Loaded {len(df_results)} rows from image_metadata table.") # Add this line

        
        
        if not df_results.empty:
            # Add STT column (1-based index)
            df_results['STT'] = range(1, len(df_results) + 1)
            
            # Format confidence for display
            df_results['ai_prediction_confidence_display'] = df_results['ai_prediction_confidence'].apply(lambda x: f"{x:.2%}")

            # Generate charts from the dataframe
            charts = generate_charts(df_results)
        else:
            # If no data, ensure charts are empty and clear old chart files
            for f in os.listdir(RESULTS_FOLDER):
                if f.endswith('.png'):
                    os.remove(os.path.join(RESULTS_FOLDER, f))
            print("No image data, cleared old chart files.")

        # Convert DataFrame to list of dicts for Jinja2
        df_rows = df_results.to_dict(orient='records')

    except Exception as e:
        print(f"Error loading image metadata or generating charts: {e}")
        flash('Lỗi khi tải dữ liệu hoặc tạo biểu đồ.', 'danger')
        df_rows = [] # Ensure df_rows is empty on error
        charts = {} # Ensure charts is empty on error

    return render_template('results.html', df_rows=df_rows, charts=charts)

@app.route('/delete_images', methods=['POST'])
@login_required # Requires login to delete images
def delete_images():
    data = request.json
    image_paths_to_delete = data.get('image_paths', [])
    delete_all = data.get('delete_all', False)

    deleted_count = 0
    try:
        if delete_all:
            # Delete all files from UPLOAD_FOLDER
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Delete all records from ImageMetadata table
            deleted_count = ImageMetadata.query.delete() # Returns number of rows deleted
            db.session.commit()
            flash('Đã xóa tất cả ảnh và dữ liệu thành công.', 'success')
        elif image_paths_to_delete:
            for img_path in image_paths_to_delete:
                # Delete file from filesystem
                if os.path.exists(img_path): # img_path is already the full path
                    os.remove(img_path)
                
                # Delete record from database
                # Find the record by image_path
                image_record = ImageMetadata.query.filter_by(image_path=img_path).first()
                if image_record:
                    db.session.delete(image_record)
                    db.session.commit()
                    deleted_count += 1
                    print(f"DEBUG: Saved image metadata for {unique_filename} to DB. Predicted: {predicted_category}, Confidence: {confidence}") # Add this line
                    #except Exception as e:
                else:
                    print(f"Warning: Image record not found for path: {img_path}")

            flash(f'Đã xóa thành công {deleted_count} ảnh và dữ liệu liên quan.', 'success')
        
        # After deleting, regenerate charts to reflect current data
        # Fetch remaining data to generate charts based on it
        remaining_images_metadata = ImageMetadata.query.all()
        if remaining_images_metadata:
            remaining_data_for_df = [
                {
                    'id': img.id,
                    'file_name': img.file_name,
                    'image_path': img.image_path,
                    'category': img.category,
                    'uploaded_by_user_id': img.uploaded_by_user_id,
                    'upload_timestamp': img.upload_timestamp,
                    'ai_predicted_category': img.ai_predicted_category,
                    'ai_prediction_confidence': img.ai_prediction_confidence,
                    'ai_top_3_predictions': img.ai_top_3_predictions
                }
                for img in remaining_images_metadata
            ]
            remaining_df = pd.DataFrame(remaining_data_for_df)
        else:
            remaining_df = pd.DataFrame() # Empty DataFrame if no images left
            
        generate_charts(remaining_df) # This will overwrite old charts with new ones

        return jsonify({'status': 'success', 'message': 'Xóa thành công.'})
    except Exception as e:
        db.session.rollback() # Rollback in case of error during deletion
        print(f"Error deleting images: {e}")
        return jsonify({'status': 'error', 'message': f'Lỗi khi xóa: {e}'}), 500


if __name__ == '__main__':
    # Initial chart generation (if data exists)
    # This block runs only once when the app starts
    with app.app_context():
        # Ensure databases are initialized before attempting to read
        # If the 'users' table existed with an 'email' column, and you remove it from the model,
        # you might need to manually drop the 'users' table in pgAdmin and let create_all() recreate it.
        # This will ensure the database schema matches the model definition.
        db.create_all()

        try:
            # Fetch data from PostgreSQL
            initial_images_metadata = ImageMetadata.query.all()
            if initial_images_metadata:
                initial_data_for_df = [
                    {
                        'id': img.id,
                        'file_name': img.file_name,
                        'image_path': img.image_path,
                        'category': img.category,
                        'uploaded_by_user_id': img.uploaded_by_user_id,
                        'upload_timestamp': img.upload_timestamp,
                        'ai_predicted_category': img.ai_predicted_category,
                        'ai_prediction_confidence': img.ai_prediction_confidence,
                        'ai_top_3_predictions': img.ai_top_3_predictions
                    }
                    for img in initial_images_metadata
                ]
                initial_df = pd.DataFrame(initial_data_for_df)
            else:
                initial_df = pd.DataFrame() # Empty DataFrame if no images
            
            generate_charts(initial_df)
        except Exception as e:
            print(f"Error during initial chart generation (PostgreSQL): {e}")
            # This might happen if the database connection is not established or tables are not created.

    app.run(debug=True, port=5000)