import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import random  
import textwrap  
  
st.set_page_config(page_title="Data Science Escape Room", page_icon="🧩", layout="wide")  
  
# --- Helper for session state ---  
def get_session():  
    if "escape_state" not in st.session_state:  
        st.session_state.escape_state = {  
            "solved": [False]*7,  
            "attempts": [0]*7,  
            "random_seed": random.randint(1, 1_000_000)  
        }  
    return st.session_state.escape_state  
  
state = get_session()  
random.seed(state["random_seed"])  
np.random.seed(state["random_seed"])  
  
# --- Room Navigation ---  
st.title("🧩 Data Science Escape Room")  
st.markdown("Solve the rooms in any order. Each room covers a key Data Science topic. Good luck!")  
  
room_names = [  
    "1. Conditional Probability",  
    "2. Association Rules",  
    "3. Linear Regression",  
    "4. Support Vector Machine",  
    "5. AUC & ROC",  
    "6. Logistic Regression",  
    "7. Autoencoder"  
]  
  
room_funcs = []  
  
# --- Room 1: Conditional Probability ---  
def room1():  
    st.header("Room 1: Conditional Probability")  
    st.write("A patient has tested positive for a rare disease. What is the probability they actually have the disease?")  
    # Randomized parameters  
    prevalence = random.randint(1, 10)  # %  
    sensitivity = random.randint(80, 99)  # %  
    specificity = random.randint(80, 99)  # %  
    st.info(f"**Prevalence:** {prevalence}%  \n**Sensitivity:** {sensitivity}%  \n**Specificity:** {specificity}%")  
    st.write("Calculate: $P(Disease | Positive)$ (as a percentage, rounded to 1 decimal place)")  
    answer = st.text_input("Your answer:", key="room1_input")  
    # Solution  
    P_D = prevalence / 100  
    P_nD = 1 - P_D  
    P_pos_D = sensitivity / 100  
    P_pos_nD = 1 - (specificity / 100)  
    P_pos = P_D * P_pos_D + P_nD * P_pos_nD  
    P_D_pos = (P_D * P_pos_D) / P_pos  
    correct = round(P_D_pos*100, 1)  
    # Pie chart  
    fig, ax = plt.subplots()  
    labels = ['TP', 'FP', 'TN', 'FN']  
    sizes = [P_D*P_pos_D, P_nD*P_pos_nD, P_nD*(1-P_pos_nD), P_D*(1-P_pos_D)]  
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')  
    st.pyplot(fig)  
    # Check answer  
    idx = 0  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if answer:  
        state["attempts"][idx] += 1  
        try:  
            user = float(answer)  
            if abs(user - correct) < 0.2:  
                st.success("Correct! You solved the room.")  
                state["solved"][idx] = True  
            else:  
                st.error("Not quite right.")  
                if state["attempts"][idx] == 1:  
                    st.info("Tip: Use Bayes' theorem!")  
                elif state["attempts"][idx] == 2:  
                    st.info(f"Tip: The answer is about {correct:.1f}%.")  
        except:  
            st.error("Please enter a number.")  
  
room_funcs.append(room1)  
  
# --- Room 2: Association Rules ---  
def room2():  
    st.header("Room 2: Association Rules")  
    st.write("You analyze supermarket data. Which product pair has the highest lift?")  
    # Randomized data  
    products = ["Bread", "Milk", "Eggs", "Butter"]  
    pairs = [("Bread", "Milk"), ("Bread", "Eggs"), ("Milk", "Eggs"), ("Butter", "Eggs")]  
    supports = np.random.uniform(0.1, 0.5, size=len(pairs))  
    lifts = np.random.uniform(0.8, 2.0, size=len(pairs))  
    df = pd.DataFrame({"Pair": [f"{a} & {b}" for a, b in pairs], "Support": supports, "Lift": lifts})  
    st.dataframe(df)  
    answer = st.selectbox("Which pair has the highest lift?", df["Pair"])  
    idx = 1  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if st.button("Submit", key="room2_btn"):  
        state["attempts"][idx] += 1  
        max_lift_pair = df.iloc[df["Lift"].idxmax()]["Pair"]  
        if answer == max_lift_pair:  
            st.success("Correct! You solved the room.")  
            state["solved"][idx] = True  
        else:  
            st.error("Not correct.")  
            if state["attempts"][idx] == 1:  
                st.info("Tip: Look for the highest number in the 'Lift' column.")  
  
room_funcs.append(room2)  
  
# --- Room 3: Linear Regression ---  
def room3():  
    st.header("Room 3: Linear Regression")  
    st.write("Fit a line to the data. Adjust the slope and intercept so the line fits best (minimizes MSE).")  
    # Random data  
    x = np.linspace(0, 10, 20)  
    true_slope = random.uniform(1, 3)  
    true_intercept = random.uniform(-2, 2)  
    noise = np.random.normal(0, 1, size=x.shape)  
    y = true_slope * x + true_intercept + noise  
    # User sliders  
    slope = st.slider("Slope", 0.0, 5.0, 1.0, 0.1)  
    intercept = st.slider("Intercept", -5.0, 5.0, 0.0, 0.1)  
    y_pred = slope * x + intercept  
    mse = np.mean((y - y_pred)**2)  
    # Plot  
    fig, ax = plt.subplots()  
    ax.scatter(x, y, label="Data")  
    ax.plot(x, y_pred, color="red", label="Your line")  
    ax.legend()  
    st.pyplot(fig)  
    st.write(f"Your MSE: {mse:.2f}")  
    idx = 2  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if mse < 1.5:  
        st.success("Great fit! You solved the room.")  
        state["solved"][idx] = True  
    elif st.button("Need a tip?", key="room3_tip"):  
        st.info(f"Try slope ≈ {true_slope:.2f}, intercept ≈ {true_intercept:.2f}")  
  
room_funcs.append(room3)  
  
# --- Room 4: Support Vector Machine ---  
def room4():  
    st.header("Room 4: Support Vector Machine")  
    st.write("Move the decision boundary to separate the classes as well as possible.")  
    # Random data  
    n = 30  
    X1 = np.random.normal([2, 2], 0.5, (n, 2))  
    X2 = np.random.normal([6, 6], 0.5, (n, 2))  
    X = np.vstack([X1, X2])  
    y = np.array([0]*n + [1]*n)  
    # User sets boundary: y = ax + b  
    a = st.slider("Slope (a)", -2.0, 2.0, 1.0, 0.1)  
    b = st.slider("Intercept (b)", -10.0, 10.0, 0.0, 0.1)  
    # Predict  
    y_pred = (X[:,1] > a*X[:,0] + b).astype(int)  
    acc = np.mean(y_pred == y)  
    # Plot  
    fig, ax = plt.subplots()  
    ax.scatter(X[:,0], X[:,1], c=y, cmap="bwr", label="Data")  
    x_line = np.linspace(X[:,0].min(), X[:,0].max(), 100)  
    ax.plot(x_line, a*x_line + b, color="green", label="Boundary")  
    ax.legend()  
    st.pyplot(fig)  
    st.write(f"Classification accuracy: {acc:.2%}")  
    idx = 3  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if acc > 0.95:  
        st.success("Excellent! You solved the room.")  
        state["solved"][idx] = True  
    elif st.button("Need a tip?", key="room4_tip"):  
        st.info("Try to make the green line go between the two clusters.")  
  
room_funcs.append(room4)  
  
# --- Room 5: AUC & ROC ---  
def room5():  
    st.header("Room 5: AUC & ROC")  
    st.write("Adjust the threshold to maximize the AUC of your classifier.")  
    # Random data  
    from sklearn.metrics import roc_curve, auc  
    y_true = np.random.randint(0, 2, 100)  
    y_scores = np.random.uniform(0, 1, 100)  
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)  
    y_pred = (y_scores > threshold).astype(int)  
    fpr, tpr, _ = roc_curve(y_true, y_scores)  
    auc_score = auc(fpr, tpr)  
    # Plot  
    fig, ax = plt.subplots()  
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")  
    ax.plot([0,1], [0,1], 'k--')  
    ax.set_xlabel("False Positive Rate")  
    ax.set_ylabel("True Positive Rate")  
    ax.legend()  
    st.pyplot(fig)  
    st.write(f"Current AUC: {auc_score:.2f}")  
    idx = 4  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if auc_score > 0.8:  
        st.success("Great! You solved the room.")  
        state["solved"][idx] = True  
    elif st.button("Need a tip?", key="room5_tip"):  
        st.info("Try different thresholds to see how the ROC curve changes.")  
  
room_funcs.append(room5)  
  
# --- Room 6: Logistic Regression (Code input) ---  
def room6():  
    st.header("Room 6: Logistic Regression")  
    st.write("Write a function that computes the sigmoid of x.")  
    st.code("def sigmoid(x):\n    # your code here")  
    code = st.text_area("Enter your function code below:", height=100, key="room6_code")  
    idx = 5  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if st.button("Check code", key="room6_btn"):  
        state["attempts"][idx] += 1  
        # Check code  
        try:  
            local_env = {}  
            exec(code, {}, local_env)  
            if "sigmoid" in local_env:  
                test = local_env["sigmoid"](0)  
                if abs(test - 0.5) < 0.01:  
                    st.success("Correct! You solved the room.")  
                    state["solved"][idx] = True  
                else:  
                    st.error("Not correct. Try again.")  
                    if state["attempts"][idx] == 1:  
                        st.info("Tip: sigmoid(x) = 1 / (1 + np.exp(-x))")  
            else:  
                st.error("Please define a function named 'sigmoid'.")  
        except Exception as e:  
            st.error(f"Error in your code: {e}")  
  
room_funcs.append(room6)  
  
# --- Room 7: Autoencoder (Slider + Image) ---  
def room7():  
    st.header("Room 7: Autoencoder")  
    st.write("Adjust the bottleneck size to reconstruct the image as well as possible.")  
    # Use a simple image (e.g. a digit)  
    from sklearn.datasets import load_digits  
    digits = load_digits()  
    img = digits.images[0]  
    bottleneck = st.slider("Bottleneck size", 1, 64, 8)  
    # Simulate reconstruction: more bottleneck = more info  
    recon = img.copy()  
    mask = np.zeros_like(img)  
    mask[:bottleneck//8, :bottleneck//8] = 1  
    recon = img * mask  
    # Show images  
    col1, col2 = st.columns(2)  
    with col1:  
        st.image(img, caption="Original", width=150)  
    with col2:  
        st.image(recon, caption="Reconstructed", width=150)  
    idx = 6  
    if state["solved"][idx]:  
        st.success("Room already solved!")  
        return  
    if bottleneck >= 8:  
        st.success("Good reconstruction! You solved the room.")  
        state["solved"][idx] = True  
    elif st.button("Need a tip?", key="room7_tip"):  
        st.info("Increase the bottleneck size for better reconstruction.")  
  
room_funcs.append(room7)  
  
# --- Room selection ---  
st.sidebar.title("Rooms")  
room_choice = st.sidebar.radio("Choose a room:", room_names)  
room_idx = room_names.index(room_choice)  
room_funcs[room_idx]()  
  
# --- Progress ---  
st.sidebar.markdown("---")  
st.sidebar.write("**Progress:**")  
for i, name in enumerate(room_names):  
    if state["solved"][i]:  
        st.sidebar.success(f"✔ {name}")  
    else:  
        st.sidebar.warning(f"✗ {name}")  
  
if all(state["solved"]):  
    st.balloons()  
    st.success("Congratulations! You have solved all rooms! 🎉")  
    st.code(f"Your escape code: DS-{state['random_seed']}")  