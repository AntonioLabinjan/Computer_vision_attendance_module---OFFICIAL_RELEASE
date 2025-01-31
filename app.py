# UMRA SAN DOK SAN NAPISA SVE KOMENTARE 
# AKO NI METODE, ONDA JE METODA GET
# iz fajla imports importamo sve (to san napravija da niman 50 linija importova na vrhu)
from imports import *

# iz fajla db startup importamo funkciju za kreiranje baze podataka i pozivamo je da se baza napravi
from db_startup import create_db
create_db()

# iz fajlova za loading modela pozivamo funkcija za loading. Učitavamo clip model za face embeddingse i model za hateful speech text multiclassification
from model_loader import load_clip_model
from load_post_check import load_model_and_tokenizer

# Load-an stvari iz env fajla (api ključeve, passworde i te sličen stvari)
load_dotenv()

# inicijaliziramo app i definiramo da je to flask aplikacija + dohvaćamo secret key iz .env (za session management)
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Config za email api (isto sve vadimo iz env)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True' # dohvaća bool varijablu iz .env i uspoređuje je s True. Ako je u .env varijabla True, ovaj check će tornat True, i pičimo dalje
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# SendGrid API Key
sendgrid_api_key = os.getenv('SENDGRID_API_KEY')

# ovo je da se ne zbrejka sve ako slučajno 2 puta iniciramo tensorflow
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# Init modela. Učitavamo dataset i weightove. Model se svaki put fine-tunea na naše nove podatke (mada to ni onaj classic fine tuning di mi mijenjamo neke parametere. On sve dela umisto nas :)
weights_path = "C:/Users/Korisnik/Desktop/cv_attendance/fine_tuned_classifier.pth"
dataset_path = "C:/Users/Korisnik/Desktop/cv_attendance/known_faces"
model, processor, classifier = load_clip_model(weights_path, dataset_path)

# In-memory storage za lica; ovo čak i ne triba spremat u bazu jer: 
                                    # a) svaki put pozovemo funkciju za loading usera kad palimo app i vjerojatno je isto brzo dali ih učita ovako, ili selektira iz baze
                                    # b) neman ni približnu ideju kako vraga se slike moru spremit u sql bazu...vjerojatno bi trebalo setupirat neki firebase samo za slike, ali dela i ovako...ako dela, ne tičen niš
# encodings su za tensore lica(1 slika => 1 tensor). Svaka klasa ima više tensora. Usporedimo tensor detected lica s tensorima koje imamo poznate
known_face_encodings = []
known_face_names = []

# Pratimo koje ljude smo logali u current sessionu. Više ga ne koristin; jer nam ne odgovara da je stvar set; ne bi bilo moguće logat istega čovika više puta u 1 sesiji (a treba mi to)
logged_names = set()


# ovo bi bilo dobro spremit u .env fajl, ali to ću ben delat (spremija san)
#app.secret_key = 'DO_NOT_VISIT_GRMIALDA' # RIJEŠENO => trebalo je bit GRIMALDA, but I misspelled it :(


# Flask-Login setup
# flask ima svoj pre-made library za login management. Samo ga instanciramo, inicijaliziramo i postavimo da ćemo login obavljat na view-u koji se zove login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Model za usera
# User je klasa. Svaki ima id, username, password i email
class User:
    # konstruktor
    def __init__(self, id, username, password, email):
        self.id = id
        self.username = username
        self.password = password
        self.email = email

    # Inače se moru dodat i složenije provjere, ali meni to ni toliko bitno. Pretpostavljamo da je svaki user aktivan, autentificiran i da ni anoniman (to su default properties čim se ulogira)
    # Flask-Login needs these properties to work correctly
    @property
    def is_active(self):
        # Return True if the user is active. You can modify this based on user status.
        return True

    @property
    def is_authenticated(self):
        # Return True if the user is authenticated
        return True

    @property
    def is_anonymous(self):
        # Return False because this is not an anonymous user
        return False

    def get_id(self):
        # Return the user's ID as a string
        return str(self.id)


# In-memory user storage (can be replaced with a database)
# users = {} OVO MI VIŠE NE TRIBA


# koristimo login manager za loading korisnika
# prosljedimo id u funkciju, spajamo se na bazu, cursor izvršava query na bazu tako da dohvaća onega korisnika koji ima odgovarajući id
@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    # kursor dohvaća prvega usera s odgovarajućim id-jem pomoću fetchone (doduše, ne more imat 2 usera, isti id, ali čisto preventivno)
    user = c.fetchone()
    # zatvaramo konekciju da se sve ne zblesira
    conn.close()

    # ako smo našli korisnika s tin id-jem, dohvaćamo atribute indeksirano po redu kako smo definirali u klasi
    if user:
        return User(id=user[0], username=user[1], password=user[2], email=user[3])
    return None

# iniciramo mail server (ovo doli je viška ja mislin, jer ima u env)
mail = Mail()


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'attendance.logged@gmail.com'  # Forši da napravin još 1 mail da bude malo više smisleno
app.config['MAIL_PASSWORD'] = 'ATTENDANCE/2025'  # Password
app.config['MAIL_DEFAULT_SENDER'] = 'attendance.logged@gmail.com'  # Adresa iz koje šaljen

mail = Mail(app)



# Function to add a known face.
# Dodaje sliku po sliku. Prosljeđuje putanju do slike i ime klase(osobe)
def add_known_face(image_path, name):
    # učitvamo sliku iz putanje koju smo prosljedili
    image = cv2.imread(image_path) 
    # ako je ni (None=> nema je; nismo je našli) dajemo value error 
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    # ako je ima, konvertiramo je u drugi format da clip more delat s njon
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # processor uzima rgb slike i pretvara ih u tensore koji su input za clip model
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        # uzimamo tensore i izvlačimo featurse iz njih tj. embeddinge
        outputs = model.get_image_features(**inputs)
    # pretvorimo u numpy array
    embedding = outputs.cpu().numpy().flatten()
    # normaliziranje za lakšu usporedbu
    known_face_encodings.append(embedding / np.linalg.norm(embedding))  
    # appendamo u listu
    known_face_names.append(name)
    print(f"Added face for {name} from {image_path}")

# Load all known faces from the 'known_faces' directory 
def load_known_faces():
    # Loop through each class (subfolder) in the 'known_faces' folder
    for student_name in os.listdir('known_faces'):
        # Path to the train subfolder for the current class => učitavamo samo train
        train_dir = os.path.join('known_faces', student_name, 'train')
        
        # Check if the train subfolder exists
        if os.path.isdir(train_dir):
            # Loop through all images in the train subfolder
            for filename in os.listdir(train_dir):
                image_path = os.path.join(train_dir, filename)
                try:
                    # Add the known face from the image
                    add_known_face(image_path, student_name)
                except ValueError as e:
                    print(e)

    # Debugging
    print(f"Loaded {len(known_face_encodings)} known face encodings")
    print(f"Known face names: {known_face_names}")


# Build an index for Facebook AI Similarity Search (FAISS) using known face encodings
def build_index(known_face_encodings):
    # Pretvaramo listu known_face_encodings u numpy array za rad s FAISS-om
    known_face_encodings = np.array(known_face_encodings)
    
    # Dobivamo dimenziju svakega embedding vektora (broj feturesa po vektoru)
    dimension = known_face_encodings.shape[1]
    
    # Kreiramo FAISS index koji koristi L2  (Euclidean distance) => korijen od kvadrirane sume razlika podudarnih elemenata u 2 vektora 
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # Dodajemo sve poznate face encodings u FAISS index
    faiss_index.add(known_face_encodings)
    
    # Vraćamo izgrađeni FAISS index
    return faiss_index


# Search for the closest face in the Faiss index
# sad koristimo faiss index za pretraživanje. uzmemo trenutni embeding koji smo ulovili na detekciji, pretražujemo faiss index da bimo našli closest match
# vraćamo closest match i klasificiramo detektirano lice kao najbližu klasu
# OVO VIŠE NE KORISTIMO
def search_face(face_embedding, faiss_index, known_face_names):
    distances, indices = faiss_index.search(face_embedding[np.newaxis, :], 1)
    if distances[0][0] < 2:  # Distance threshold for recognition (bilo je 0.6)
        return known_face_names[indices[0][0]]
    return "Unknown"

# pozivamo funkciju za učitavanje svih lica/embeddinga
load_known_faces()

# pretvorimo lica u numpy array za bit sigurni u buildamo index
known_face_encodings = np.array(known_face_encodings)  # Ensure encodings are a numpy array
faiss_index = build_index(known_face_encodings)


# Initialize the webcam. Stavimo nulu za koristit defaultnu kameru (ako smo toliko luksuzni da imamo više kamera, moremo stavit 1,2,3 itd. ;)
video_capture = cv2.VideoCapture(0)

# Face detection using Haar Cascade
# ovo ubacimo da detektira lica i crta bounding boxeve oko njih
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicijaliziranje atributa za trenutnu prisutnost
current_subject = None
attendance_date = None
start_time = None
end_time = None


# Password validation
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>/]", password):
        return "Password must contain at least one special character."
    return None

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # get je jer dohaćamo formular, post je šaljemo zahtjev za signup
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        repeat_password = request.form['repeat_password']
        email = request.form['email']
        
        # standard validation
        if password != repeat_password:
            flash("Passwords do not match. Please try again.", "error")
            return redirect(url_for('signup'))

        password_error = validate_password(password)
        if password_error:
            flash(password_error, "error")
            return redirect(url_for('signup'))

        try:
            conn = sqlite3.connect('attendance.db')
            c = conn.cursor()

            c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            if c.fetchone() is not None:
                flash("Username or email already taken. Please choose a different one.", "error")
                conn.close()
                return redirect(url_for('signup'))

            # Hashiranje
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            # Save to db
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                      (username, hashed_password, email))
            conn.commit()
            conn.close()

            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash("An error occurred during signup. Please try again.", "error")
            return redirect(url_for('signup'))

    return render_template('signup.html')


# ovo samo uzima naš input u login formu i provjerava dali se podudara s nekin usernameon i dali se hashed passwordi poklapaju
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()  # Fetches user row (dohvati 1 usera)
        conn.close()

        if user:
            # User is found
            user_id, db_username, db_password, db_email = user
            if check_password_hash(db_password, password):
                # Successful login
                login_user(User(id=user_id, username=db_username, password=db_password, email=db_email))
                return redirect(url_for('index'))

        flash("Invalid username or password")
        return redirect(url_for('login'))

    return render_template('login.html')

# test za thunderclient; niš bitno
@app.route('/hello', methods=['GET'])
def hello():
    return "Hello, World!"

@app.route('/logout')
@login_required
def logout():
    # samo call-a ugrađenu funkciju za logout
    logout_user()
    return redirect(url_for('login'))

@app.route('/frontend_camera')
def fcc():
    return render_template('frontend_camera_cap.html')


@app.route('/set_subject', methods=['GET', 'POST'])
@login_required # moramo bit ulogirani za pristupit ruti
def set_subject():
    # ove varijable su globalne jer želimo da ih se mora čitat/dohvaćat i van funkcije (tribat će nan kad budemo logali attendance)
    global current_subject, attendance_date, start_time, end_time
    # ako šaljemo post request, spremamo u varijable ovo ča smo prosljedili kroz req. form
    if request.method == 'POST':
        current_subject = request.form['subject']
        attendance_date = request.form['date']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        # renderiramo template s dodanima podacima
        return render_template('set_subject_success.html', 
                               subject=current_subject, 
                               date=attendance_date, 
                               start_time=start_time, 
                               end_time=end_time)
    return render_template('set_subject.html')

# provjera dali je trenutno vrijeme unutar intervala koji smo definirali za predavanje
def is_within_time_interval():
    # u now spremamo trenutni datum i vrime
    now = datetime.now()
    # iz now extractamo posebno datum i posebno vrime
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")
    # vraća true ako smo unutar intervala, odnosno false ako nismo
    return (current_date == attendance_date and 
            start_time <= current_time <= end_time)

# path do dataseta
dataset_path = "c:/Users/Korisnik/Desktop/cv_attendance/known_faces"

# ruta za dodavanje studenta 
@app.route('/add_student', methods=['GET', 'POST'])
# moramo bit ulogirani
@login_required
def add_student():
    # dohvaćamo ime iz requesta i izlistavamo sve slike koje smo poslali kroz request (1 je minimum, al more i više)
    if request.method == 'POST':
        name = request.form['name']
        images = request.files.getlist('images')
        
        # Novi subfolder za nove studente (4each student se dela folder)
        student_dir = os.path.join('known_faces', name)
        os.makedirs(student_dir, exist_ok=True)
        for image in images:
            # Dodajemo svaku sliku u studentov folder
            image_path = os.path.join(student_dir, image.filename)
            image.save(image_path)
            # pozivamo add_known_face za svaku sliku
            add_known_face(image_path, name)
        
        # dodajemo dinamički parametar name u template za success
        return render_template('add_student_success.html', name=name)

    return render_template('add_student.html')

# Route to confirm success => samo izrenderiramo stranicu ako uspije
@app.route('/add_student_success')
def add_student_success():
    return render_template('add_student_success.html')

# Ako known faces slučajno ne postoje, napravimo ih
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

'''
# OVO NI VALJALO; A IZGLEDALO JE BAŠ DOBRO :(
# Function to add a face encoding from a live feed frame
def add_known_face_from_frame(image_frame, name):
    global known_face_encodings, known_face_names

    # Convert frame to RGB and process it
    image_rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    # Generate the image feature embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Normalize and add to encodings
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))
    known_face_names.append(name)
    print(f"Added face for {name} from live capture")
'''
# Route to capture live feed images and add a new student
# Uglavnon, ovo je isti vrag kao ono static dodavanje, samo lovimo iz kamere

def add_known_face_from_frame(image_frame, name):
    # koristimo one iste encodinge i nameove od prije
    global known_face_encodings, known_face_names

    # Convert frame to RGB and process it
    image_rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    # Generate the image feature embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Normalize the embedding
    embedding = outputs.cpu().numpy().flatten()
    normalized_embedding = embedding / np.linalg.norm(embedding)
    
    # If `known_face_encodings` is a list, append directly
    if isinstance(known_face_encodings, list):
        known_face_encodings.append(normalized_embedding)
    else:
        # If it's a numpy array, use numpy.vstack to add the new embedding
        known_face_encodings = np.vstack([known_face_encodings, normalized_embedding])

    # Append the name
    known_face_names.append(name)
    print(f"Added face for {name} from live capture")

# Route to capture live feed images and add a new student
# Isto kao i ono static dodavanje, samo lovimo slike preko kamere
@app.route('/add_student_live', methods=['GET', 'POST'])
def add_student_live():
    if request.method == 'POST':
        name = request.form['name']
        student_dir = os.path.join(dataset_path, name)
        os.makedirs(student_dir, exist_ok=True)
        
        # Start webcam capture
        cap = cv2.VideoCapture(0)
        frame_count = 0
        required_frames = 25  # Number of frames to capture => 25 po klasi/osobi (more više, more manje)
        # otpri camera capture i lovi frameove sve dok ih više nema (ugasimo kameru ili ima dovoljno frejmova); onda samo udri break
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Display live video feed 
            cv2.imshow("Capture Face - Press 'q' to Quit", frame)

            # Capture every frame => sve dok ih je manje od 25 (required)
            if frame_count < required_frames:
                # Save frame in student's directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # => dohvati datum i vrime za timestamp za naziv slike
                frame_path = os.path.join(student_dir, f"{name}_{timestamp}_{frame_count}.jpg") # nazovi sliku ime_timstamp_redni_broj_framea
                cv2.imwrite(frame_path, frame)
                
                # Process and save the embedding = > spremi tensor i povećaj frame count za 1
                add_known_face_from_frame(frame, name)
                frame_count += 1
            
            # Killaj petlju ako ima dovoljno slika ili smo quitali s q
            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= required_frames:
                break

        # oslobodi kameru, zapri detection window
        cap.release()
        cv2.destroyAllWindows()

        # isto ko manualno dodavanje
        return redirect(url_for('add_student_success', name=name))
    
    return render_template('add_student_live.html')


# koristimo onaj mail server koji smo definirali gore
def send_attendance_notification(name, date, time, subject): # => ovi podaci nas zanimaju (postavili smo ih kroz set subject i ubacit ćemo ih u mail)
    message = Mail(
        # izgled maila
        from_email='attendance.logged@gmail.com', 
        to_emails='alabinjan6@gmail.com', # napravit neki official mail za ovaj app da ne koristin svoj mail
        subject=f'Attendance Logged for {name}', 
        plain_text_content=f'Attendance for {name} in {subject} on {date} at {time} was successfully logged.'
    )
    
    try:
        print("Attempting to send email...")
        #sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        response = sg.send(message)
        # debug ako sendgrid ne dela
        print(f"Email sent: {response.status_code}")
        print(f"Response body: {response.body}")  
    except Exception as e:
        print(f"Error sending email: {str(e)}")



# Load Haar cascadesa za face detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global varijable za attendance ( ovo je viška, već iman gori)
current_subject = None
attendance_date = None
start_time = None
end_time = None



def detect_face(frame):
    """Detect faces in the frame using Haar cascades."""
    # Convert the frame (which is in BGR color) to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Haar Cascade classifier detektira lica
    # `scaleFactor` skalira veličinu da svako detektirano lice bude cca isto 
    # `minNeighbors` osigurava da svako lice mora bit u bar 5 frameova da bi ga priznalo 
    # `minSize` osigurava minimalnu veličinu
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # vraća koordinate svakega lica pomoću bounding boxa
    return faces


def detect_eye_movement(frame, face):
    return True
'''
    # lovimo koordinate lica i oči. Ako ulovi 2 oka, pretpostavlja da smo čovik
    (x, y, w, h) = face
    roi_gray = frame[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Check if two eyes are detected for liveness
    return len(eyes) >= 2  # Live face detected (eyes are open)
'''
# provjerava dali trepćemo
def detect_eye_blink(face, gray_frame):
    return True
'''
    (x, y, w, h) = face
    roi_gray = gray_frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    return len(eyes) < 2  # Eyes are closed if fewer than 2 eyes are detected
'''
# provjerava dali mičemo glavu (dali ima neke razlike u koordinatama između frameova)
def detect_head_movement(old_gray, new_gray, faces):
    return True
'''
    for face in faces:
        (x, y, w, h) = face
        roi_old = old_gray[y:y+h, x:x+w]
        roi_new = new_gray[y:y+h, x:x+w]
        difference = cv2.absdiff(roi_old, roi_new)
        non_zero_count = np.count_nonzero(difference)
        if non_zero_count > 50:  # Threshold for movement
            return True
    return False
'''
liveness_frame_count = 0 # init varijable
LIVENESS_FRAMES_THRESHOLD = 5  # Number of consecutive frames for liveness confirmation

# za svako lice provjeravamo dali je pomicalo oči i glavu; ako je, povećamo liveness frame count
def check_liveness_over_time(frame, faces, old_gray, new_gray):
    return True
'''
    global liveness_frame_count
# delaj provjere za sva detektirana lica   
    for face in faces:
        # varijable za treptanje i micanje glave (more bit true/false)
        eyes_closed = detect_eye_blink(face, new_gray)
        head_moved = detect_head_movement(old_gray, new_gray, faces)
        

        if not eyes_closed and head_moved:  # oba uvjeta moraju bit zadovoljena da se frame prizna ko live frame
            liveness_frame_count += 1
        else:
            liveness_frame_count = 0  
    
    if liveness_frame_count >= LIVENESS_FRAMES_THRESHOLD:
        return True  # ako liveness pasa u dovoljno frameova, smatramo osobu živom
    return False
'''
async def check_liveness(frame, faces):
    return True
'''
    # za svako detektirano lice provjeravamo liveness. Stavljeno je u async s ciljen da dela bar malo brže
    for face in faces:
        if detect_eye_movement(frame, face):
            return True  # Live face 
    return False  # Spoof 
'''
# e ovo je zabavno
def log_attendance(name, frame):
    # ovo je globalna varijable jer smo je postavili u drugoj funkciji (set subject i dohvaćamo je tu). Svako loganje attendancea će bit za točno taj subject ako je postavljen
    global current_subject, attendance_date, start_time, end_time
    # ako nismo postavili predmet ili ako nismo unutar intervala (starttime > x && x < endtime)
    if current_subject is None or not is_within_time_interval():
        print("Subject is not set or current time is outside of allowed interval. Attendance not logged.")
        return frame

    # lovimo trenutni datum i vrijeme
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create datetime object for start time => kad počinje loganje attendancea
    start_time_obj = datetime.strptime(f"{attendance_date} {start_time}", "%Y-%m-%d %H:%M")

    # Toleriramo do 15 minuta kašnjenja (starttime + interval od 14 minuta)
    late_time_obj = start_time_obj + timedelta(minutes=14)

    # Check if the current time is late => dali je trenutno vrijeme veće od starttimea za više od late_time_obj(a.k.a 14 minuta)
    if now > late_time_obj:
        # za pisat po camera captureu; niš bistro
        cv2.putText(frame, f"Late Entry: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # takamo se na bazu i inicijaliziramo kursor
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # provjeravamo dali već postoji u tablici attendance zapis za to ime, taj datum i taj predmet i fetchamo prvi zakov zapis
    c.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND subject = ?", (name, date, current_subject))
    existing_entry = c.fetchone()

    # ako postoji, vraćamo alert da je prisutnost već loggana
    if existing_entry:
        print(f"Attendance for {name} on {date} for subject {current_subject} is already logged.")
        return frame

    # ako ne postoji , pozivamo cursor insert i dodamo zapis
    # Late is 1 if we are late, if not then 0 => bool flag za provjeru
    c.execute("INSERT INTO attendance (name, date, time, subject, late) VALUES (?, ?, ?, ?, ?)", 
              (name, date, time, current_subject, 1 if now > late_time_obj else 0))
    conn.commit()
    conn.close()

    print(f"Logged attendance for {name} on {date} at {time} for subject {current_subject}.")

    # poziva se funkcija za slanje maila
    send_attendance_notification(name, date, time, current_subject)

    return frame

def perform_liveness_check(frame):
    """Capture video from the camera and perform liveness detection."""
    cap = cv2.VideoCapture(0)  # Use the primary camera
    live_face_detected = True # varijable je po defaultu false, pa je hitimo na true ako detektiramo živost

    # opet ona fora da detecta, sve dok više ne detecta
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False

    # Detect faces in the current frame
    faces = detect_face(frame)

    # If faces are detected, check for liveness
    for face in faces:
        is_alive = check_liveness(frame, [face])
        (x, y, w, h) = face

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box

        # dajemo labelu i print
        if is_alive:
            live_face_detected = True
            print("OK, real")
            cv2.putText(frame, "Live Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print("ERROR, fake")
            cv2.putText(frame, "Spoof Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cap.release()

    # daje true/false i o temu ovisi dal ćemo ga spremit ili ne
    return live_face_detected


def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.6): #bilo je k1=3,k2=5
    """
    Classifies a face embedding using majority voting logic.
    Args:
        face_embedding: The embedding of the face to classify.
        faiss_index: FAISS index for known faces.
        known_face_names: List of known face names corresponding to FAISS index.
        k1: Number of nearest neighbors for majority voting.
        k2: Number of fallback neighbors.
        threshold: Similarity threshold for classification.
    Returns:
        majority_class: Predicted class label.
        class_counts: Vote counts for each class.
    """
    D, I = faiss_index.search(face_embedding[np.newaxis, :], max(k1, k2))
    votes = {}

    for idx, dist in zip(I[0], D[0]):
        if idx == -1 or dist > threshold:
            continue
        label = known_face_names[idx]
        votes[label] = votes.get(label, 0) + 1

    if votes:
        majority_class = max(votes, key=votes.get)
    else:
        majority_class = "Unknown"

    return majority_class, votes


# COMPUTER VISION MAGIJA
def generate_frames(k1=20, k2=5, threshold=0.7):
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    old_gray = None  # Initialize for liveness detection

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't grab frame. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)

        if old_gray is None:
            old_gray = gray.copy()

        # Display liveness instructions
        cv2.putText(frame, "Move your head around", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if check_liveness_over_time(frame, faces, old_gray, gray):
            cv2.putText(frame, "Liveness Confirmed", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                inputs = processor(images=face_rgb, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()
                face_embedding /= np.linalg.norm(face_embedding)  # Normalize embedding

                if len(known_face_encodings) > 0:
                    # Perform majority vote classification
                    majority_class, class_counts = classify_face(
                        face_embedding, faiss_index, known_face_names, k1, k2, threshold
                    )

                    # Format vote results for display
                    match_text = f"{majority_class} ({class_counts[majority_class]} votes)"
                    if majority_class != "Unknown":
                        frame = log_attendance(majority_class, frame)
                else:
                    majority_class = "Unknown"
                    match_text = "Unknown (0 votes)"

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, match_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Liveness Failed: No Movement Detected", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        old_gray = gray.copy()

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()




# na ruti za video feed se poziva generate frames funkcija
@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') # http live streaming di je svaki frame zasebni response



# / ruta. Početna stranica
@app.route('/')
def index():
    # dohvatimo api ključ iz env
    api_key = os.getenv('WEATHER_API_KEY')
    location = "Pula"  # Ili nešto drugo
    # pozivamo funkciju za get_weather_forecast
    weather_condition = get_weather_forecast(api_key, location)
    
    # Logika za vremensku prognozu => ako dohvaćena prognoza sadrži neki bad keyword, predictamo da će bit loše u suprotnemu, će bit ok
    if predict_absence_due_to_weather(weather_condition):
        message = "Bad weather predicted, late entries due to traffic problems are possible."
    else:
        message = "No significant weather issues expected. Students should come on time."
    
    # Provjera je li korisnik već vidio popup => kad se prvi prvi prvi put ulogiramo, dobit ćemo popup o privacy-ju
    if 'seen_privacy_policy' not in session:
        show_popup = True
        session['seen_privacy_policy'] = False  # Po defaulutu je false, ako ga nismo vidili (logično :D)
    else:
        show_popup = False
    
    return render_template('index.html', weather_condition=weather_condition, message=message, show_popup=show_popup)


# GET ruta za dohvat zapisa o prisutnosti
@app.route('/attendance', methods=['GET'])
@login_required
def attendance():
    # Lovimo filter parametre iz request argumenta => ako ga nema u request argumentu, onda je taj parametar prazan
    name_filter = request.args.get('name')
    subject_filter = request.args.get('subject')
    date_filter = request.args.get('date')
    weekday_filter = request.args.get('weekday')
    month_filter = request.args.get('month')
    year_filter = request.args.get('year')
    late_filter = request.args.get('late')

    # connectamo se na bazu i upalimo kursor
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Dynamic build of query
    query = "SELECT rowid, subject, name, date, time, late FROM attendance WHERE 1=1"
    params = []
    # ako smo dali određeni argument, onda appendamo taj argument u tekst querija i u parametre
    if name_filter:
        query += " AND name = ?"
        params.append(name_filter)
    
    if subject_filter:
        query += " AND subject = ?"
        params.append(subject_filter)
    
    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    
    if weekday_filter:
        query += " AND strftime('%w', date) = ?"
        params.append(weekday_filter)
    
    if month_filter:
        query += " AND strftime('%m', date) = ?"
        params.append(f"{int(month_filter):02d}")
    
    if year_filter:
        query += " AND strftime('%Y', date) = ?"
        params.append(year_filter)
    
    if late_filter:
        query += " AND late = ?"
        params.append(late_filter)

    query += " ORDER BY date, time"
    c.execute(query, params)
    
    # dohvatimo sve rezultate koji odgovaraju filteru
    records = c.fetchall()
    conn.close()

    # Group records by date and subject  => dictionary za recordse po datumu. Svaki datum ima predmete za keyeve a values si svi studenti koji su bili tamo
    grouped_records = {}
    for rowid, subject, name, date, time, late in records:
        if date not in grouped_records:
            grouped_records[date] = {}
        if subject not in grouped_records[date]:
            grouped_records[date][subject] = []
        grouped_records[date][subject].append((rowid, name, time, late))
    
    return render_template('attendance.html', grouped_records=grouped_records)


# preuzimanje podataka
@app.route('/download')
def download_attendance():
    # spajamo se na bazu i dohvaćamo sve attendance podatke
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT subject, name, date, time FROM attendance ORDER BY subject, date, time")
    records = c.fetchall()
    conn.close()

    # Create string buffer za pisat u csv
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Initialize previous subject and student count
    previous_subject = None
    student_count = 0

    # Headeri csv-a
    writer.writerow(['Subject', 'Name', 'Date', 'Time', 'Number of students'])

    # Grouped CSV data by subject
    for record in records:
        subject, name, date, time = record
        
        if subject != previous_subject:
            # Write the student count for the previous subject, if exists
            if previous_subject is not None:
                writer.writerow(['', '', '', '', student_count])  # student count row
                writer.writerow([])  # empty row between each subject
            # Write the new subject name and reset student count
            writer.writerow([subject])  # subject header
            previous_subject = subject
            student_count = 0

        # Write the student's attendance record
        writer.writerow(['', name, date, time])
        student_count += 1  # Increment student count for this subject

    # Write student count for the last subject
    if previous_subject is not None:
        writer.writerow(['', '', '', '', student_count])

    # Seek to the beginning of the stream
    output.seek(0)
    
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=attendance.csv"})

# Ista stvar ko ovo gore, samo se još pošalje na mail

@app.route('/download_and_email')
def download_and_email_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT subject, name, date, time FROM attendance ORDER BY subject, date, time")
    records = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    previous_subject = None
    student_count = 0

    writer.writerow(['Subject', 'Name', 'Date', 'Time', 'Number of students'])

    for record in records:
        subject, name, date, time = record

        if subject != previous_subject:
            if previous_subject is not None:
                writer.writerow(['', '', '', '', student_count])  
                writer.writerow([])  
            writer.writerow([subject]) 
            previous_subject = subject
            student_count = 0

        writer.writerow(['', name, date, time])
        student_count += 1 

    if previous_subject is not None:
        writer.writerow(['', '', '', '', student_count])

    output.seek(0)

    csv_data = output.getvalue()

    # dohvatimo request argumente
    user_email = request.args.get('email') 

    return render_template('download.html', csv_data=csv_data, user_email=user_email)
 
# brisanje po id-ju => spojimo se na bazu, pošaljemo query 
@app.route('/delete_attendance/<int:id>', methods=['POST'])
def delete_attendance(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE rowid = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('attendance'))

# dohvat statistike=> obični sql upiti
@app.route('/statistics')
@login_required
def statistics():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT name, subject, COUNT(*) FROM attendance GROUP BY name, subject")
    student_attendance = c.fetchall()

    c.execute("SELECT subject, COUNT(*) FROM attendance GROUP BY subject")
    subject_attendance = c.fetchall()

    conn.close()
    return render_template('statistics.html', student_attendance=student_attendance, subject_attendance=subject_attendance)

# initially debug ruta, ali san je pustija => dohvat svih distinct studenata koji postoje
@app.route('/students')
@login_required
def students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT name FROM attendance ORDER BY name")
    students = c.fetchall()
    conn.close()

    return render_template('students.html', students=students)

# Ove rute su sve za grafikone => fetchaš podatke iz baze, malo ih obradiš i plotaš s matplotlibon
# BITNO!!! NE NANKA POKUŠAVAT OPIRAT PLOTOVE AKO JOŠ NEMA ZABILJEŽENIH PRISUTNOSTI, JER ĆE SE ZBREJKAT
@app.route('/plot/student_attendance')
def plot_student_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT name, COUNT(*) as count FROM attendance GROUP BY name")
    data = c.fetchall()
    conn.close()

    # Prepare data for plotting
    names, counts = zip(*data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=names, palette='viridis')
    plt.title('Attendance by Student')
    plt.xlabel('Number of Attendances')
    plt.ylabel('Student')

    # Save the plot to a BytesIO object and return it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

@app.route('/plot/subject_attendance')
def plot_subject_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT subject, COUNT(*) as count FROM attendance GROUP BY subject")
    data = c.fetchall()
    conn.close()

    # Prepare data for plotting
    subjects, counts = zip(*data)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=subjects, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Attendance by Subject')

    # Save the plot to a BytesIO object and return it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')


@app.route('/plot/monthly_attendance')
def plot_monthly_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT date, COUNT(*) as count FROM attendance GROUP BY date")
    data = c.fetchall()
    conn.close()

    # Prepare data for plotting
    dates, counts = zip(*data)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=dates, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Monthly Attendance Distribution')

    # Save the plot to a BytesIO object and return it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

@app.route('/plots')
@login_required
def plots():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute('SELECT COUNT(*) FROM attendance')
    attendance_count = c.fetchone()[0]  # Get the count from the result
    conn.close()

    # Provjera dali ima podataka; ako nema, ne otvarat !!
    if attendance_count > 0:
        return render_template('plot_router.html')
    else:
        flash("No attendance records found. Please add some attendance data before viewing the plots, because everything will break if you try to plot non-existing data <3", "error")
        return render_template('flash_redirect.html')  # Render a new template for displaying the message

        




# API CALL...AKO POKAŽE ODREĐENO VRIME, DAMO ALERT DA BI MOGLI KASNIT

# use api url, get response, turn response to json, return data
def get_weather_forecast(api_key, location="your_city"):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=1"
    response = requests.get(url)
    data = response.json()
    return data["forecast"]["forecastday"][0]["day"]["condition"]["text"]

# provjera dali sadrži bad weather keywordse i vraćanje pripadne poruke
def predict_absence_due_to_weather(weather_condition):
    bad_weather_keywords = ["rain", "storm", "snow", "fog", "hurricane", "blizzard","tornado","sandstorm"]
    for keyword in bad_weather_keywords:
        if keyword in weather_condition.lower():
            return True
    return False

# dohvatimo forecast, ubacimo forecast u analiziranje bad weathera i vratimo poruku (json)
@app.route('/predict_absence', methods=['GET'])
def predict_absence():
    api_key = "fe2e5f9339b2434db60124446241408"
    location = "London"
    weather_condition = get_weather_forecast(api_key, location)
    
    if predict_absence_due_to_weather(weather_condition):
        message = "Bad weather predicted, late entries due to traffic problems are possible."
    else:
        message = "No significant weather issues expected. Students should come on time"
    
    return jsonify({
        "weather_condition": weather_condition,
        "message": message
    })

# učitavanje NLP modela za filtriranje neprimjerenih riječi
tokenizer, model2 = load_model_and_tokenizer()
# prima message, rastavlja je na tokene i pretvori u tensor (threshold označava koliko je osjetljiv)
# vraća koji je probbability da je poruka offensive. Ako je offensiveness veći od 30%, flag-a poruku i ne prikaže je
def is_inappropriate_content(message, threshold=0.30):
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model2(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    offensive_score = probs[0][1].item()  #label index 1 corresponds to "offensive" class
    return offensive_score > threshold


# OVO JE KAO NEKI PROFESORSKI FORUM/CHAT ILI ČA JA ZNAN KAKO BI SE TO ZVALO
# Niš pametno; običan sql upit
@app.route('/announcements', methods=['GET'])
@login_required
def announcements():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM announcements ORDER BY date_time ASC")
    announcements = c.fetchall()
    conn.close()
    return render_template('announcements.html', announcements=announcements)

@app.route('/post_announcement', methods=['POST'])
# samo šaljemo post s tekstom poruke, ubacimo ga u model i vidimo dali se more objavit. Ako da, sprema se u bazu i prikaže. Autor je trenutno ulogirani korisnik
@login_required
def post_announcement():
    message = request.form['message']
    if not message:
            return jsonify({"error": "Message cannot be empty."}), 400
    if is_inappropriate_content(message):
        return redirect(url_for('announcements', warning="Your announcement contains inappropriate language and was not posted."))
    
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    teacher_name = current_user.username

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO announcements (date_time, teacher_name, message) VALUES (?, ?, ?)",
              (date_time, teacher_name, message))
    conn.commit()
    conn.close()
    return redirect(url_for('announcements'))

# šaljemo post request s delete upiton po id-ju
@app.route('/delete_announcement/<int:id>', methods=['POST'])
@login_required
def delete_announcement(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM announcements WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('announcements'))


# Pošaljemo get, dohvatimo poruku po id-ju. Uredimo je, pasa kroz model da vidimo dali je neprimjerena i editiramo
@app.route('/announcements/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_announcement(id):

    if request.method == 'POST':
        data = request.get_json()  # Expecting JSON data for PUT request
        message = data.get('message')
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        teacher_name = current_user.username

        if not message:
            return jsonify({"error": "Message cannot be empty."}), 400

        # Check for inappropriate content
        if is_inappropriate_content(message, threshold=0.5):
            return jsonify({"error": "Announcement contains inappropriate content."}), 400

        try:
            with sqlite3.connect('attendance.db') as conn:
                c = conn.cursor()
                c.execute("""
                    UPDATE announcements 
                    SET date_time = ?, message = ? 
                    WHERE id = ? AND teacher_name = ?
                """, (date_time, message, id, teacher_name))
                if c.rowcount == 0:  # If no rows are updated
                    return jsonify({"error": "Announcement not found or not authorized."}), 404

                return jsonify({"success": "Announcement updated successfully."}), 200
        except sqlite3.Error as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500

    elif request.method == 'GET':
        try:
            with sqlite3.connect('attendance.db') as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM announcements WHERE id = ?", (id,))
                announcement = c.fetchone()

                if announcement is None or announcement[2] != current_user.username:
                    return redirect(url_for('announcements'))

                return render_template('edit_announcement.html', announcement=announcement)
        except sqlite3.Error as e:
            flash(f"Error retrieving announcement: {str(e)}", "error")
            return redirect(url_for('announcements'))



#generira izvještaj o prisutnosti studenata za svaki predmet. 
# Iz baze podataka izračunava ukupan broj održanih predavanja, postotak prisutnosti svakog studenta, te prosječnu prisutnost za predmet.
#  Studenti se također označavaju kao oni koji ispunjavaju ili ne ispunjavaju prag od 50% prisutnosti.
#  Generirani izvještaj prosljeđuje se predlošku attendance_report.html za prikaz.
@app.route('/report')
@login_required
def report():
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute('SELECT DISTINCT subject FROM attendance')
    subjects = cur.fetchall()

    report = []

    for subject in subjects:
        subject_name = subject['subject']

        cur.execute('SELECT COUNT(DISTINCT date) as total_lectures FROM attendance WHERE subject = ?', (subject_name,))
        total_lectures = cur.fetchone()['total_lectures']

        cur.execute('''SELECT name, COUNT(*) as attended_lectures, 
                       (COUNT(*) * 100.0 / ?) as attendance_percentage 
                       FROM attendance 
                       WHERE subject = ? 
                       GROUP BY name 
                       ORDER BY attendance_percentage DESC''', 
                   (total_lectures, subject_name))
        students_attendance = cur.fetchall()

        students_with_status = []
        for student in students_attendance:
            meets_requirement = student['attendance_percentage'] >= 50
            students_with_status.append({
                'name': student['name'],
                'attended_lectures': student['attended_lectures'],
                'attendance_percentage': student['attendance_percentage'],
                'meets_requirement': meets_requirement
            })

        cur.execute('''SELECT AVG(attendance_percentage) as avg_attendance 
                       FROM (SELECT COUNT(*) * 100.0 / ? as attendance_percentage 
                             FROM attendance 
                             WHERE subject = ? 
                             GROUP BY name)''', 
                   (total_lectures, subject_name))
        avg_attendance = cur.fetchone()['avg_attendance']

        report.append({
            'subject': subject_name,
            'total_lectures': total_lectures,
            'average_attendance': avg_attendance,
            'students': students_with_status
        })

    conn.close()

    return render_template('attendance_report.html', report=report)


# Ruta koja će dohvatit sva kašnjenja, i prikazat grafikon da se vidi kad najviše kasne
# Koristimo collections/counter za brojanje najčešćih sati i dana kad ljudi kasne
@app.route('/late_analysis', methods=["GET", "POST"])
@login_required
def late_entries():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Fetch all late entries
    c.execute("SELECT * FROM attendance WHERE late = 1")
    late_entries = c.fetchall()
    conn.close()

    # Init counters
    hour_counter = Counter()
    weekday_counter = Counter()

    for entry in late_entries:
        time_in = entry[2]  
        date = entry[1]     

        # Convert time and date 
        time_obj = datetime.strptime(time_in, "%H:%M:%S")
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        # Count the hour
        hour_counter[time_obj.hour] += 1
        
        # Count the weekday (0=Monday, 6=Sunday)
        weekday_counter[date_obj.weekday()] += 1

    # Convert results to lists => rendering
    most_common_hour = hour_counter.most_common(1)[0] if hour_counter else None
    most_common_weekday = weekday_counter.most_common(1)[0] if weekday_counter else None

    # Show all hours from 00 to 23
    hours = list(range(24))
    hour_counts = [hour_counter.get(hour, 0) for hour in hours]  # Get count or 0 if not in the counter

    # Visualization
    if hour_counter:
        plt.bar(hours, hour_counts)
        plt.xticks(hours)  # Ensure all hours are labeled on the x-axis
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Late Entries')
        plt.title('Late Entries by Hour')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
    else:
        plot_url = None

    return render_template('late_entries.html', 
                           late_entries=late_entries, 
                           most_common_hour=most_common_hour, 
                           most_common_weekday=most_common_weekday,
                           plot_url=plot_url)


# WEBSCRAPING ROUTES => BEUTIFUL SOUP ZA STRANICE I PDFPLUMBER ZA PDF-OVE


def scrape_github_profile(url):
    try:
        # http req na github profil (moj url)
        response = requests.get(url)
        response.raise_for_status()

        # parsiramo cijelu stranicu u html-u
        soup = BeautifulSoup(response.text, 'html.parser')

        # tražimo dio s usernameon
        name = soup.find('span', class_='p-name').text.strip()

        # tražimo dio s biography
        bio = soup.find('div', class_='p-note user-profile-bio mb-3 js-user-profile-bio f4').text.strip() if soup.find('div', class_='p-note user-profile-bio mb-3 js-user-profile-bio f4') else 'No bio available'

        # tražimo profile picture
        profile_picture = soup.find('img', class_='avatar-user')['src'] if soup.find('img', class_='avatar-user') else None

        # tražimo broj dollowera
        followers = soup.find('span', class_='text-bold').text.strip()

        # tražimo broj public repozitorija
        repositories_element = soup.find('span', class_='Counter')
        repositories = repositories_element.text.strip() if repositories_element else '0'  # => hendlanje slučaja ako ih ima 0

        # vraćanje scrapeanih podataka
        return {
            'name': name,
            'bio': bio,
            'profile_picture': profile_picture,
            'followers': followers,
            'repositories': repositories,
        }
    except Exception as e:
        print(f"Error scraping the website: {e}")
        return None


# ruta za scraping koja koristi gornju funkciju
@app.route('/scrape_github', methods=['GET'])
def github_profile():
    # koji url?
    url = 'https://github.com/AntonioLabinjan'
    
    if not url:
        return jsonify({"error": "Please provide a GitHub profile URL"}), 400
    
    profile_info = scrape_github_profile(url)
    
    if profile_info:
        # ako smo nešto našli, izrenderirat ćemo stranicu s ovim contentom
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GitHub Profile</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #111; 
                    color: #f9f9f9;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #222; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    text-align: center;
                }}
                h1 {{
                    color: #ff6600; 
                }}
                p {{
                    font-size: 18px;
                    line-height: 1.6;
                }}
                .followers, .repositories {{
                    font-weight: bold;
                    color: #ff6600; 
                    font-size: 20px;
                }}
                .profile-picture {{
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    border: 2px solid #ff6600;
                    margin-bottom: 20px;
                }}
                .back-button {{
                    margin-top: 20px;
                    padding: 10px 20px;
                    font-size: 18px;
                    color: #fff;
                    background-color: #ff6600;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                .back-button:hover {{
                    background-color: #cc5200;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <img src="{profile_info['profile_picture']}" alt="Profile Picture" class="profile-picture">
                <h1>{profile_info['name']}</h1>
                <p><strong>Bio:</strong> {profile_info['bio']}</p>
                <p class="followers">Followers: {profile_info['followers']}</p>
                <p class="repositories">Repositories: {profile_info['repositories']}</p>
                <button class="back-button" onclick="window.location.href='/'">Go to Home</button>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_content), 200
    else:
        return jsonify({"error": "Failed to scrape the GitHub profile"}), 500




# scraping za pdf-ove
def extract_pdf_text(pdf_url):
    # dajemo request za dohvat pdf-a , ugasimo ssl
    response = requests.get(pdf_url, verify=False)
    # ako je response ok, skinemo fajl
    if response.status_code == 200:
        with open("calendar.pdf", "wb") as f:
            f.write(response.content)
        
        # Extract texta iz fajla
        text_content = ""
        with pdfplumber.open("calendar.pdf") as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        return text_content
    else:
        return "Failed to retrieve PDF."

# filter za praznike
def get_non_working_days(text):
    # Define keywords
    keywords = [
        "Blagdan", "Praznik", "nenastavni", "odmor", "Božić", "Nova Godina", 
        "Tijelovo", "Dan sjećanja", "Uskrs", "Svi sveti", "Sveta tri kralja", 
        "Dan državnosti", "Velike Gospe", "Dan domovinske zahvalnosti"
    ]
    
    # tražimo datume s tima keywordsima
    non_working_days = []
    # dohvatimo sve dane koji sadrže neki keyword, splitamo svaki sa /n
    for line in text.split("\n"):
        if any(keyword in line for keyword in keywords):
            non_working_days.append(line.strip())
    # onda ih izjoinamo
    return "\n".join(non_working_days)

# Flask route to display the filtered non-working days
@app.route("/calendar")
def show_calendar():
    pdf_url = "https://www.unipu.hr/_download/repository/Sveu%C4%8Dili%C5%A1ni%20kalendar%20za%202024._2025..pdf"
    calendar_text = extract_pdf_text(pdf_url)
    non_working_days = get_non_working_days(calendar_text)
    
    # Splitamo extractane dane
    non_working_days_list = non_working_days.split("\n")

    # izrenderiramo ih s bullet pointsima
    html_content = f"""
    <html>
        <head>
            <title>Non-Working Days 2024/2025</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f0f0f0;
                    color: #333;
                }}
                h1 {{
                    text-align: center;
                    color: #0056b3;
                }}
                ul {{
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                    line-height: 1.6;
                    font-size: 14px;
                    list-style-type: disc;
                }}
                ul li {{
                    margin-bottom: 10px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    background-color: #ffffff;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Non-Working Days 2024/2025</h1>
                <ul>
                    {''.join(f"<li>{day}</li>" for day in non_working_days_list if day.strip())}
                </ul>
            </div>
        </body>
    </html>
    """
    return render_template_string(html_content)



'''
Run the flask app on port 5144
'''
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5145))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    #app.run(host="0.0.0.0", port=5145, debug=True, use_reloader = False)
   # app.run(debug=True, use_reloader=False)
