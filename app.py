import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

def main():
    st.title("Real-time Sobel and Laplacian Edge Detection")
    st.write("This app applies Sobel and Laplacian filters to webcam input.")

    # Sidebar for parameters
    st.sidebar.header("Filter Parameters")
    
    # Kernel size (must be odd)
    kernel_sizes = [3, 5, 7, 9, 11]
    kernel_size = st.sidebar.select_slider(
        "Kernel Size",
        options=kernel_sizes,
        value=3
    )
    
    # Sobel parameters
    st.sidebar.subheader("Sobel Parameters")
    sobel_scale = st.sidebar.slider("Sobel Scale", 1, 20, 1)
    sobel_delta = st.sidebar.slider("Sobel Delta", 0, 255, 0)
    sobel_ddepth = cv2.CV_16S  # Fixed for better results
    
    # Direction for Sobel
    sobel_direction = st.sidebar.radio(
        "Sobel Direction",
        ["Combined", "X-Direction", "Y-Direction"]
    )
    
    # Laplacian parameters
    st.sidebar.subheader("Laplacian Parameters")
    laplacian_scale = st.sidebar.slider("Laplacian Scale", 1, 20, 1)
    laplacian_delta = st.sidebar.slider("Laplacian Delta", 0, 255, 0)
    laplacian_ddepth = cv2.CV_16S  # Fixed for better results
    
    # Display settings
    display_mode = st.sidebar.radio(
        "Display Mode",
        ["Side by Side", "Up and Down"]
    )
    
    # Preprocessing options
    st.sidebar.subheader("Preprocessing")
    apply_blur = st.sidebar.checkbox("Apply Gaussian Blur", True)
    blur_kernel_size = st.sidebar.slider("Blur Kernel Size", 1, 15, 5, step=2)
    
    # Start/Stop camera button
    start_button = st.sidebar.button("Start Camera", key="start_button")
    stop_button_placeholder = st.sidebar.empty()
    stop_button_pressed = False
    
    # Create placeholders for the video frames
    if display_mode == "Side by Side":
        col1, col2, col3 = st.columns(3)
        original_frame_placeholder = col1.empty()
        sobel_frame_placeholder = col2.empty()
        laplacian_frame_placeholder = col3.empty()
    else:  # Up and Down
        original_frame_placeholder = st.empty()
        col1, col2 = st.columns(2)
        sobel_frame_placeholder = col1.empty()
        laplacian_frame_placeholder = col2.empty()
    
    if start_button:
        # Start webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Please check your camera connection.")
            return
        
        # Show stop button
        stop_button_pressed = stop_button_placeholder.button("Stop Camera", key="stop_button0")
        
        while not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break
                
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur if selected
            if apply_blur:
                gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
            
            # Apply Sobel filter
            if sobel_direction == "X-Direction":
                sobel_x = cv2.Sobel(gray, sobel_ddepth, 1, 0, ksize=kernel_size, scale=sobel_scale, delta=sobel_delta)
                sobel_result = cv2.convertScaleAbs(sobel_x)
            elif sobel_direction == "Y-Direction":
                sobel_y = cv2.Sobel(gray, sobel_ddepth, 0, 1, ksize=kernel_size, scale=sobel_scale, delta=sobel_delta)
                sobel_result = cv2.convertScaleAbs(sobel_y)
            else:  # Combined
                sobel_x = cv2.Sobel(gray, sobel_ddepth, 1, 0, ksize=kernel_size, scale=sobel_scale, delta=sobel_delta)
                sobel_y = cv2.Sobel(gray, sobel_ddepth, 0, 1, ksize=kernel_size, scale=sobel_scale, delta=sobel_delta)
                abs_sobel_x = cv2.convertScaleAbs(sobel_x)
                abs_sobel_y = cv2.convertScaleAbs(sobel_y)
                sobel_result = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            
            # Apply Laplacian filter
            laplacian = cv2.Laplacian(gray, laplacian_ddepth, ksize=kernel_size, scale=laplacian_scale, delta=laplacian_delta)
            laplacian_result = cv2.convertScaleAbs(laplacian)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sobel_rgb = cv2.cvtColor(sobel_result, cv2.COLOR_GRAY2RGB)
            laplacian_rgb = cv2.cvtColor(laplacian_result, cv2.COLOR_GRAY2RGB)
            
            # Update placeholders with new frames
            original_frame_placeholder.image(frame_rgb, caption="Original", use_container_width=True)
            sobel_frame_placeholder.image(sobel_rgb, caption="Sobel Filter", use_container_width=True)
            laplacian_frame_placeholder.image(laplacian_rgb, caption="Laplacian Filter", use_container_width=True)
            
            # Check if the stop button is pressed
            
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Release the webcam
        cap.release()
        st.write("Camera stopped")

if __name__ == "__main__":
    main()