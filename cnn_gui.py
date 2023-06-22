import numpy as np
import tkinter as tk
import keras.utils as image
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import model_from_json




def browse_image():
    filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if filename:
        image = Image.open(filename)
        image.save("image.jpg")  # Save as "image.jpg" in root directory
        success_label.config(text="Image saved successfully!", fg="green")
        display_image("image.jpg")

def run_code():
    output_text.delete("1.0", tk.END)
    # load json and create model
    json_file = open('classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("classifier.h5")
    test_image = image.load_img('image.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    # training_set.class_indices
    if result[0][0] == 1:
        prediction = 'It is a Dog'
    else:
        prediction = 'It is a Cat'
    output = prediction  
    output_text.insert(tk.END, output)

def display_image(filename):
    image = Image.open(filename)
    image = image.resize((300, 200))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

root = tk.Tk()
root.title("Cat or Dog Classifier")
root.geometry("600x500")  
root.configure(bg="light gray")

additional_text = tk.Label(root, text="Upload an image of either a dog or a cat")
additional_text.pack()

browse_button = tk.Button(root, text="Browse", command=browse_image, bg="gray", fg="white")
browse_button.pack(pady=10)

success_label = tk.Label(root, text="", fg="green", bg="light gray")
success_label.pack(pady=5)

image_label = tk.Label(root)
image_label.pack(pady=10)

code_button = tk.Button(root, text="Click Here", command=run_code, bg="gray", fg="white")
code_button.pack(pady=10)

additional_text = tk.Label(root, text="Click the button to check if image is of dog or cat")
additional_text.pack()

output_text = tk.Text(root, bg="white", height=8)
output_text.pack(pady=10)

root.mainloop()
