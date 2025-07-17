**SIMPLIFYSORT-Sorting Algorithm Visualizer**

This is a web application designed to visually demonstrate how various fundamental sorting algorithms work, step by step. Users can select an algorithm, input their own data or generate random data, and then observe an animated visualization of the sorting process, highlighting key operations like comparisons and swaps.
Features

●	Algorithm Selection: Choose from a variety of sorting algorithms:
○	Bubble Sort
○	Insertion Sort
○	Merge Sort
○	Quick Sort
○	Selection Sort
○	Radix Sort

●	Custom Input: Enter your own comma-separated list of numbers or use the "Randomize Data" button.
●	Interactive Visualization: Array elements are represented as bars. The height of each bar corresponds to its value.
●	Step-by-Step Controls: Navigate through the sorting process with "Next" and "Previous" buttons.
●	Animation Playback: "Play" and "Pause" controls for automatic animation.
●	Adjustable Speed: A slider to control the visualization speed.
●	Visual Feedback: Bars change color to indicate their role in the current step (comparing, swapping, sorted, active pivot, Radix pass, bucket element).
●	Educational Messages: A status message provides context for each step of the algorithm.

Technologies Used
●	Backend:
○	Python 3.x
○	Flask: A lightweight web framework for the API.
○	Flask-CORS: For handling Cross-Origin Resource Sharing, enabling communication between the frontend and backend.
●	Frontend:
○	HTML5: For the structure of the web page.
○	CSS3: For styling and animations, primarily using custom CSS and variables for the pastel pink theme.
○	Tailwind CSS (CDN): A utility-first CSS framework for rapid UI development and responsiveness.
○	JavaScript (Vanilla JS): For all interactive logic, DOM manipulation, and API communication.

Project Structure
The project consists of two main files:
your_project_folder/
├── app.py                  # Python Flask Backend
├── index.html              # HTML/CSS/JavaScript Frontend
└── requirements.txt        # Python dependencies

How to Run the Application:
Follow these steps to get the sorting algorithm visualizer up and running on your local machine.

Prerequisites
●	Python 3.x installed on your system.
●	pip (Python package installer)

1. Clone or Download the Project
First, get the project files onto your computer. If you have them already, proceed to the next step.

2. Install Backend Dependencies
Open your terminal or command prompt, navigate to your project directory (where app.py and requirements.txt are located), and install the necessary Python packages:
pip install -r requirements.txt

3. Run the Flask Backend
In the same terminal window (or a new one, ensuring you are in the project directory), start the Flask development server:
python app.py

4. Open the Frontend
Now, open the index.html file in your web browser:
●	Option A (Simple): Navigate to the project folder in your file explorer and double-click index.html. It will open in your default browser.
●	Option B (Recommended for Development - using VS Code Live Server):
1.	If you use VS Code, install the "Live Server" extension.
2.	In VS Code's Explorer, right-click on index.html.
3.	Select "Open with Live Server".
   
This will open the page in your browser, usually at http://127.0.0.1:5500/index.html, and provides live reloading for convenience during development.
The frontend (index.html) will automatically connect to the backend (app.py) running on http://127.0.0.1:5000.
