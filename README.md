Biometric Voting System
A secure web-based voting system with biometric authentication, candidate management, and user feedback collection. Built with Flask, it supports voter authentication, voting with location tracking, feedback submission, and an admin dashboard for managing candidates and viewing results.
Features

User Authentication: Secure login with username and biometric data.
Voting: Users can vote for candidates with geolocation capture.
Feedback Collection: Post-voting feedback saved to feedback.json.
Admin Dashboard: View voting results, candidate details, vote locations, and user feedback.
Responsive UI: Glassmorphism design with Font Awesome icons and Poppins font.
Skip Feedback: Users can skip feedback submission after voting.

Prerequisites

Python 3.8+
Flask
A modern web browser (Chrome, Firefox, etc.)
Internet connection for CDN-hosted assets (Font Awesome, Google Fonts)

Installation

Clone the Repository:
git clone https://github.com/your-username/biometric-voting-system.git
cd biometric-voting-system


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install flask


Initialize Data Files:

Ensure feedback.json exists in the project root:[]


Set file permissions (Linux/Mac):chmod 666 feedback.json




Run the Application:
python app.py


Access at http://localhost:5000.



File Structure
biometric-voting-system/
├── app.py                  # Main Flask application
├── templates/
│   ├── feedback.html       # Feedback submission page
│   ├── admin_dashboard.html # Admin dashboard for results and management
│   └── (other templates)   # login.html, vote.html, etc.
├── feedback.json           # Stores user feedback
└── README.md               # Project documentation

Usage
User Flow

Login: Navigate to /login, enter username, and authenticate at /authenticate.
Vote: Select a candidate at /vote (geolocation captured).
Feedback: Provide feedback at /feedback or click "Skip" to return to /.
View Feedback: Feedback is saved in feedback.json.

Admin Flow

Admin Login: Go to /admin (default: username admin, password admin123).
Dashboard: Access /admin_dashboard to:
View voting results (total voters, votes per candidate).
Manage candidates (add, edit, delete).
See vote locations and user feedback.


Add Candidate: Click "Add Candidate" to register new candidates.

Dependencies

Python Packages:
flask: Web framework
logging: For debug/error logs
json: For handling feedback.json
os: For file operations
time: For timestamp generation


External Assets:
Font Awesome 6.0.0 (via CDN)
Google Fonts (Poppins)



Troubleshooting

"Failed to save feedback" Error:

Check Flask logs for details (e.g., ERROR: Failed to write feedback.json: Permission denied).
Ensure feedback.json exists and is writable:chmod 666 feedback.json


Verify feedback.json is in the project root and initialized with [].


404 on Routes:

Ensure Flask is running (python app.py).
Check URLs (e.g., /feedback, /admin_dashboard).


Feedback Not Displaying:

Confirm feedback.json contains valid JSON.
Check /admin_dashboard route in app.py loads feedback.json.


Session Errors:

Ensure users vote before accessing /feedback (sets session['username']).
Admins must log in at /admin before accessing /admin_dashboard.



Contributing

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a Pull Request.

License
MIT License
Copyright (c) 2025 Manasa G
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
