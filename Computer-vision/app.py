# --------------------------#
# Author : Hritik Jaiswal
# Github : https://github.com/hritik5102
# Repository : https://github.com/hritik5102/Fundamentals_of_DS_ML_DL
# --------------------------#

# To save your changes, copy your custom theme into the clipboard and paste it into the[theme] section of your .streamlit/config.toml file.
# [theme]
# primaryColor="#f63366"
# backgroundColor="#FFFFFF"
# secondaryBackgroundColor="#f0f2f6"
# textColor="#262730"
# font="sans serif"


# Import package
import streamlit as st
from PIL import Image
import cv2
import os

st.set_page_config(page_title='Fundamentals of Computer vision',
                   layout="wide", page_icon="🚀")


def main():

    # list of folder name
    list_of_folder_name = [i for i in sorted(os.listdir('./'))]

    # split folder names into num and names
    spliter = [i.split(" - ") for i in list_of_folder_name]
    nums, name = [], []

    # collect number is one list and name in others
    for i in range(1, len(spliter)):
        if len(spliter[i]) == 1:
            continue
        nums.append(spliter[i][0])
        name.append(spliter[i][1])

    # concatenate welcome in list of folders name
    list_of_folder_name = ["Welcome"] + list_of_folder_name[1:]

    st.sidebar.title("Content")
    selected_box = st.sidebar.selectbox(
        "Choose one of the following",
        list_of_folder_name[:-2]
    )

    if selected_box == list_of_folder_name[0]:
        welcome(nums, name)
    if selected_box == list_of_folder_name[1]:
        image_01()
    if selected_box == list_of_folder_name[2]:
        video_02()
    if selected_box == list_of_folder_name[3]:
        matplotlib_03()
    if selected_box == list_of_folder_name[4]:
        drawing_function_04()
    if selected_box == list_of_folder_name[5]:
        masking_05()
    if selected_box == list_of_folder_name[6]:
        trackbar_06()
    if selected_box == list_of_folder_name[7]:
        blurring_07()


def welcome(nums, name):
    st.markdown('''
    # Computer Vision Tutorial &nbsp; ![](https://img.shields.io/github/forks/hritik5102/Fundamentals_of_DS_ML_DL?style=social) ![](https://img.shields.io/github/stars/hritik5102/Fundamentals_of_DS_ML_DL?style=social) ![](https://img.shields.io/github/watchers/hritik5102/Fundamentals_of_DS_ML_DL?style=social)

    ![](https://img.shields.io/github/repo-size/hritik5102/Fundamentals_of_DS_ML_DL) ![](https://img.shields.io/github/license/hritik5102/Fundamentals_of_DS_ML_DL?color=red)    [![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/hritik5102/Fundamentals_of_DS_ML_DL)
    ![](https://img.shields.io/github/issues/hritik5102/Fundamentals_of_DS_ML_DL?color=green) ![](https://img.shields.io/github/issues-pr/hritik5102/Fundamentals_of_DS_ML_DL?color=green) ![](https://img.shields.io/github/downloads/hritik5102/Fundamentals_of_DS_ML_DL/total) ![](https://img.shields.io/github/last-commit/hritik5102/Fundamentals_of_DS_ML_DL) ![](https://img.shields.io/github/contributors/hritik5102/Fundamentals_of_DS_ML_DL)

    Computer Vision is one of the hottest topics in artificial intelligence. It is making tremendous advances in self-driving cars, robotics as well as in various photo correction apps. Steady progress in object detection is being made every day. GANs is also a thing researchers are putting their eyes on these days. Vision is showing us the future of technology and we can’t even imagine what will be the end of its possibilities.

    So do you want to take your first step in Computer Vision and participate in this latest movement? Welcome you are at the right place. From this article, we’re going to have a series of tutorials on the basics of image processing and object detection. This is the first part of OpenCV tutorial for beginners and the complete set of the series is as follows:

    # Clone git repository

    ```bash
        $ git clone "https://github.com/hritik5102/Fundamentals_of_DS_ML_DL"
    ```

    You can run and edit the algorithms or contribute to them using [Gitpod.io](https://www.gitpod.io/), a free online development environment, with a single click.

    [![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](http://gitpod.io/#https://github.com/hritik5102/Fundamentals_of_DS_ML_DL)

    ''')

    html_temp = """
    <h1>Content Gallary ✨</h1>
    <hr/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    code_link = ["[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/01%20-%20Image)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/02%20-%20Video)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/03%20-%20Matplotlib)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/04%20-%20Drawing%20Shapes)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/05%20-%20Masking)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/06%20-%20Trackbar)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/07%20-%20Blurring%20And%20Smoothing)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/08%20-%20Morphological_Transformation)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/09%20-%20Thresholding)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/10%20-%20Adaptive%20thresolding)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/11%20-%20Otsu%20thresolding)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/12%20-%20Transformation)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/13%20-%20Callbacks)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/14%20-%20Canny%20edge%20detection)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/15%20-%20Sobel%20and%20Laplacian%20Operation)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/16%20-%20Gradient)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/17%20-%20Histogram)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/18%20-%20Hough%20Line)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/19%20-%20Fourier%20transform)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/20%20-%20Contour)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/21%20-%203d%20map)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/22%20-%20Background-filter)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/23%20-%20Corner%20detection)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/24%20-%20Gamma%20Correction)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/25%20-%20Stereo_blend_2_Camera)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/26%20-%20Template%20Matching)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/27%20-%20Haar%20Cascade)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/28%20-%20Color%20tracking)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/29%20-%20Mouse%20Movement)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/30%20-%20Color%20Detection)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/31%20-%20Jack_in_the_box)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/32%20-%20Lane%20detection)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/33%20-%20Pillow%20Library)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/34%20-%20Digital%20Negative%20and%20Gray%20level%20slicing)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/35%20-%20Intensity%20transformation)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/36%20-%20Histogram%20Equalization)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/37%20-%20Cartoon_Effect)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/38%20-%20Image%20Registration%20Using%20Homography)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/39%20-%20Convert%20URL%20to%20image)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/40%20-%20OpenCV%20In%20Colab)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/41%20-%20Word%20Detection)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/42%20-%20Mixing%202%20Frames)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/43%20-%20Color_Slicing)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/44%20-%20Camera%20Calibration)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/45%20-%20Align_images)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/46%20-%20Sketch)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/47%20-%20Handling%20Mouse%20Events)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/48%20-%20Invisible%20Cloak)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/49%20-%20Region_of_interest)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/50%20-%20Image_Addition_and_Blending)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/51%20-%20Bitwise%20Operations)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/52%20-%20Skin%20lesion%20detection)",
                 "[![](https://img.shields.io/badge/Code-Python-blue)](https://github.com/hritik5102/Fundamentals_of_DS_ML_DL/tree/master/Computer-vision/53%20-%20Object%20Counter)"

                 ]

    left, center, right = st.columns((2, 3, 1))

    with left:
        left.markdown('''**No.** ''', unsafe_allow_html=True)
        for i in nums:
            left.write(i)
    with center:
        center.markdown('''**Discription**''', unsafe_allow_html=True)
        for i in name:
            center.write(i)
    with right:
        right.markdown('''**Code**''', unsafe_allow_html=True)
        for link in code_link:
            right.markdown(link, unsafe_allow_html=True)

    st.markdown('''
        # License

    Licensed under the [MIT License](LICENSE)

    # Contributing to Computer vision Guide

    All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome.

    A detailed overview on how to contribute can be found in the contributing guide. There is also an overview on GitHub.

    If you are simply looking to start working with the voicenet codebase, navigate to the GitHub "issues" tab and start looking through interesting issues. There are a number of issues listed under Docs and good first issue where you could start out.

    You can also triage issues which may include reproducing bug reports, or asking for vital information such as version numbers or reproduction instructions.

    Or maybe through using you have an idea of your own or are looking for something in the documentation and thinking ‘this can be improved’. You can do something about it!

    Feel free to ask questions on the mailing list or on Slack.

    # Contributor
    ''')
    html_temp = """

    |                                                                                                                                                                                                                   <a href="https://hritik5102.github.io/"><img src="https://avatars.githubusercontent.com/hritik5102" width="150px" height="150px" /></a>                                                                                                                                                                                                                    |
    | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
    |                                                                                                                                                                                                                                                             **[Hritik Jaiswal](https://linktr.ee/hritikdj)**                                                                                                                                                                                                                                                              |
    | <a href="https://twitter.com/imhritik_dj"><img src="https://i.ibb.co/kmgQVyW/twitter.png" width="32px" height="32px"></a> <a href="https://github.com/hritik5102"><img src="https://cdn.iconscout.com/icon/free/png-256/github-108-438008.png" width="32px" height="32px"></a> <a href="https://www.facebook.com/hritik.jaiswal.56808"><img src="https://i.ibb.co/zmYNW4p/facebook.png" width="32px" height="32px"></a> <a href="https://www.linkedin.com/in/hritik-jaiswal-22a136166/"><img src="https://i.ibb.co/Kx2GSrT/linkedin.png" width="32px" height="32px"></a> |
    """
    st.markdown(html_temp, unsafe_allow_html=True)


def image_01():
    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Reading an image</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Read an image
    img = cv2.imread('Images and Videos/dog.png')

    # Convert RGB to Grayscale
    gray = cv2.imread('Images and Videos/dog.png', 0)

    st.markdown('''
      When the image file is read with the OpenCV function ```imread()```, the order of colors is ```BGR (blue, green, red)```.
      On the other hand, in Pillow, the order of colors is assumed to be ```RGB (red, green, blue)```.

      When reading a color image file, OpenCV ```imread()``` reads as a NumPy array ndarray of row ```(height) x column (width) x color (3)```.
      The order of color is ```BGR (blue, green, red)```.
      ''', unsafe_allow_html=True)

    st.code('''img = cv2.imread('../Images and Videos/dog.png')''',
            language='python')

    st.markdown('''The OpenCV function ```imwrite()``` that saves an image assumes that the order of colors is BGR, so it is saved as a correct image.''', unsafe_allow_html=True)

    code = '''cv2.imwrite('Save_Dog.png', img)'''
    st.code(code, language='javascript')

    st.markdown('''Convert RGB to Grayscale(1 Channel)''',
                unsafe_allow_html=True)
    code = '''gray = cv2.imread('../Images and Videos/dog.png', 0)'''
    st.code(code, language='javascript')

    st.markdown(
        '''But when displaying the image, it show the image as RGB image instead BGR''', unsafe_allow_html=True)
    code = '''cv2.imshow('BGR_Image', img)'''
    st.code(code, language='javascript')

    if st.button('See Original Image'):
        original = Image.open('Images and Videos/dog.png')
        placeholder = st.image(original, use_column_width=True)
        if st.button("Hide original image"):
            placeholder.empty()

    left, center, right = st.columns(3)

    with left:
        html_temp = """
        <h2 style="text-align:center;font-weight:bold">BGR Image</h2>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image(img, width=500, use_column_width=True)

    with center:
        html_temp = """
        <h2 style="text-align:center;font-weight:bold">RGB Image</h2>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 width=500, use_column_width=True)

    with right:
        html_temp = """
        <h2 style="text-align:center;font-weight:bold">Grayscale Image</h2>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        st.image(gray, width=500, use_column_width=True)

    # parameters
    #   1. Shape - Height, Width, Color
    #   2. dtype - type

    st.success(f'Shape of the Original image: {img.shape}')
    st.success(f'Shape of the Grayscale image: {gray.shape}')
    st.success(f'dtype of the image: {img.dtype}')
    st.success(f'type of the image: {type(img)}')

# ffmpeg -i Original_frame.avi -vcodec libx264 flip_frame.mp4


def video_02():
    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Capturing a video from a webcam or from video file</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("Original frame")
    video_file = open('02 - Video/demo_output.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4', start_time=0)

    st.header("Hue saturation value - HSV")
    video_file = open('02 - Video/hsv.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4', start_time=0)

    st.header("Blue color detected")
    video_file = open('02 - Video/res.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4', start_time=0)

    st.header("Flip Operation")
    video_file = open('02 - Video/flip_frame.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4', start_time=0)


def matplotlib_03():

    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Exploring Image operation in Matplotlib</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    html_temp = """
                <h2 style="font-weight:bold">Display a sample image using Matplotlib</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_1.png', use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Annotate image using Matplotlib</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    # with st.beta_expander("🧙 Click here to view the image 🔮"):
    st.image('03 - Matplotlib/Figure_2.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Display a Grayscale image using Matplotlib</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_3.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Display a Digital negative image using Matplotlib</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_4.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Display an 24-bit RGB image</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_6.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Display individual 8bit (Red, Green, Blue) image</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_6.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Display image when Colormap is set to "hot"</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_7.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Display image when Colormap is set to "nipy_spectral"</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_8.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Color scale reference</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_9.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Histogram plot - To Define the thresold</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_10.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Original image Vs Contrast enhanced image</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_11.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Expand contrast by cliping upper end of histogram</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_12.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Interpolation = "bilinear"[pixelated]</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_13.png', width=400, use_column_width=True)

    html_temp = """
                <h2 style="font-weight:bold">Interpolation = "nearest"</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.image('03 - Matplotlib/Figure_14.png', width=400, use_column_width=True)
    html_temp = """
                <h2 style="font-weight:bold">Interpolation = "bicubic"</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('03 - Matplotlib/Figure_15.png', width=400, use_column_width=True)


def drawing_function_04():
    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Exploring Drawing function in OpenCV</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    html_temp = """
                <h2 style="font-weight:bold">Draw a line</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_1.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Draw a rectangle</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    # with st.beta_expander("🧙 Click here to view the image 🔮"):
    st.image('04 - Drawing Shapes/Figure_2.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Fill the rectangle</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_3.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Draw a circle</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_4.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Draw a ellipse</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_6.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Draw a polygon</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_6.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Original image for testing</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('Images and Videos/image8.jpg', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Draw a line on an image</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_7.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Draw a Rectangle on an image</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_8.png', width=500)

    html_temp = """
                <h2 style="font-weight:bold">Put a text on an image</h2>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image('04 - Drawing Shapes/Figure_9.png', width=500)


def masking_05():
    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Masking operation in OpenCV</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("Example 1 : Extract Blue Color from OpenCV Logo")
    left, right = st.columns(2)
    with left:
        html_temp = """
                    <h3 style="font-weight:bold">Sample image : OpenCV Logo</h3>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('Images and Videos/opencv-logo.png', width=300)

    with right:
        html_temp = """
                    <h3 style="font-weight:bold">HSV Image</h3>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_1.png', width=300)

    left, right = st.columns(2)

    with left:
        html_temp = """
        <h3 style="font-weight:bold">Masking blue region</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_2.png', width=300)

    with right:
        html_temp = """
        <h3 style="font-weight:bold">Extract Blue Color from Image</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        st.image('05 - Masking/Figure_3.png', width=300)

    left, center, right = st.columns(3)
    with center:
        html_temp = """
        <h3 style="text-align:center;font-weight:bold">Convert background from black to white</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_4.png', width=300)

    st.header("Example 2 : Rectangular masking")
    left, center, right = st.columns(3)
    with center:
        html_temp = """
        <h3 style="text-align:center;font-weight:bold">Sample image : Messi</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('Images and Videos/target.jpg', width=300)

    left, right = st.columns(2)

    with left:
        html_temp = """
        <h3 style="font-weight:bold">Construct a rectangular mask</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_5.png', width=300)

    with right:
        html_temp = """
        <h3 style="font-weight:bold">Extract Messi from the Image</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_6.png', width=300)

    st.header("Example 3 : Circular masking")
    left, center, right = st.columns(3)
    with center:
        html_temp = """
        <h3 style="text-align:center;font-weight:bold">Sample image : Earth</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('Images and Videos/earth.jpg', width=300)

    left, right = st.columns(2)

    with left:
        html_temp = """
        <h3 style="font-weight:bold">Construct a circular mask</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_7.png', width=300)

    with right:
        html_temp = """
        <h3 style="font-weight:bold">Extract Earth from the Image</h3>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.image('05 - Masking/Figure_8.png', width=300)


def trackbar_06():
    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Creating a trackbar</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("Example 1")
    st.video('06 - Trackbar/Figure_1.mp4', format='video/mp4', start_time=0)

    st.header("Example 2")
    st.video('06 - Trackbar/Figure_2.mp4', format='video/mp4', start_time=0)


def video_render(header, path):
    """
    Input : Header : Header name 
            Path : Source image/video path

    Output : Render on streamlit app
    """
    st.subheader(header)
    st.video(path, format="video/mp4", start_time=0)


def image_render(header, path):
    """
    Input : header : Designed header
            Path : Source image/video path

    Output : Render on streamlit app
    """
    html_temp = f'<h3 style="font-weight:bold">{header}</h3>'
    st.markdown(html_temp, unsafe_allow_html=True)
    st.image(path, width=300)


def image_render_column_width(header, path):
    """
    Input : Header : Header name 
            Path : Source image/video path

    Output : Render on streamlit app
    """
    st.subheader(header)
    st.image(path, use_column_width=True)


def blurring_07():

    html_temp = """
    <div style="background-color:#02203c;padding:10px">
    <h2 style="color:white;text-align:center;font-weight:bold">Smoothing and Blurring Techniques in Image Processing</h2>
    </div>
    <br/><br/>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_render_column_width('Linear image filtering',
                              '07 - Blurring And Smoothing/Figure_14.png')
    video_render('Applying Linear image filtering on different values of K & L',
                 '07 - Blurring And Smoothing/Figure_15.mp4')
    image_render_column_width(
        '2D Convolution ( Image Filtering )', '07 - Blurring And Smoothing/Figure_1.png')
    image_render_column_width(
        'Custom kernel operation', '07 - Blurring And Smoothing/Figure_2.png')
    image_render_column_width(
        'Implementation of Gaussian filter using 2D Convolution', '07 - Blurring And Smoothing/Figure_3.png')
    image_render_column_width(
        'Averaging/Mean filtering operation on different kernel size', '07 - Blurring And Smoothing/Figure_4.png')
    image_render_column_width(
        'Boxfiltering filtering operation on different kernel size', '07 - Blurring And Smoothing/Figure_5.png')
    video_render('Applying Average/mean filtering on different kernel size',
                 '07 - Blurring And Smoothing/Figure_6.mp4')
    image_render_column_width(
        'Gaussian filtering operation on different kernel size', '07 - Blurring And Smoothing/Figure_7.png')
    video_render('Applying Gaussian filtering on different kernel size, sigma X and Y',
                 '07 - Blurring And Smoothing/Figure_8.mp4')
    image_render_column_width(
        'Median filtering operation on salt and pepper noise image', '07 - Blurring And Smoothing/Figure_9.png')
    video_render('Applying Median filtering on different kernel size',
                 '07 - Blurring And Smoothing/Figure_10.mp4')
    image_render_column_width(
        'Bilateral filtering operation', '07 - Blurring And Smoothing/Figure_11.png')
    image_render_column_width(
        '5 Iteration of Bilateral filtering', '07 - Blurring And Smoothing/Figure_12.png')
    video_render('Applying Bilateral filtering on different Diameter, Sigma Color and Sigma Space',
                 '07 - Blurring And Smoothing/Figure_13.mp4')
    image_render_column_width(
        'Add Gaussian noise to an image', '07 - Blurring And Smoothing/Figure_16.png')
    image_render_column_width(
        'Add Salt and Pepper noise to an image', '07 - Blurring And Smoothing/Figure_17.png')
    image_render_column_width(
        'Add Poission noise to an image', '07 - Blurring And Smoothing/Figure_18.png')
    image_render_column_width(
        'Add Speckle noise to an image', '07 - Blurring And Smoothing/Figure_19.png')


if __name__ == '__main__':
    main()
