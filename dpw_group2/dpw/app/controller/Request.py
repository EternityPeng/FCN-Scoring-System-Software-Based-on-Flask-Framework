import base64
from app.models.AssignQ import AssignQ
from app.models.LLM_Answer import LLM_Answer
from flask import Blueprint, Flask, request, render_template, jsonify, Response, redirect, url_for, session
import io
from PIL import Image
from app.models.LLM_Answer_Imgae import LLM_Answer_Image
from app.models.Request import Request
from app.models.Request_image import Request_image
from app.models.base import db
from app.models.course import Course
from app.models.PersonalExperiment import PersonalExperiment


from flask import Flask, request, session, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import os
import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
import io

RequestBP = Blueprint('Request', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# Function to create bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# Function to create Pascal label colormap
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3
    return colormap

# Function to convert label to color image
def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

# Load the pre-trained model and modify it
pretrained_net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
net.transpose_conv.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 64))
loaded_model = net
loaded_model.load_state_dict(torch.load('model50.pth'))

# Function to process image and get prediction
def process_image(image):
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    input = image_transform(image).unsqueeze(0)

    output = loaded_model(input)
    prediction = output.argmax(dim=1).squeeze().cpu()
    color_mask = label_to_color_image(prediction.numpy())
    color_mask = Image.fromarray(color_mask.astype('uint8'))
    alpha = 0.5

    if image.mode != 'RGB':
        image = image.convert('RGB')
    if color_mask.mode != 'RGB':
        color_mask = color_mask.convert('RGB')

    color_mask = color_mask.resize(image.size, resample=Image.NEAREST)
    combined_img = Image.blend(image, color_mask, alpha)
    return combined_img


# @RequestBP.route('/Request', methods=['GET', 'POST'])
# def create_assignq():
#     courses = Course.query.all()  # 获取所有课程
#     if request.method == 'POST':
#         qtext = request.form['qtext']
#         course_id = request.form.get('course_id')  # 从表单获取课程 ID
#         llm_used = request.form['llm_used']
#         score = int(request.form['score'])
#         comment = request.form['comment']
#
#         new_request = Request(
#             qtext=qtext,
#             course_id=course_id,  # 保存课程 ID
#             llm_used=llm_used,
#             score=score,
#             comment=comment,
#             llm_answer_id=None,
#             new_score=None,
#             explanation=None,
#             user_id=None,
#             course_number=None,
#             course_name=None,
#             course_category=None,
#             request_type='create_assignq'
#         )
#         db.session.add(new_request)
#         db.session.flush()
#
#         request_id = new_request.request_id
#
#         files = request.files.getlist('images')  # 获取所有上传的图片文件
#         for file in files:
#             if file and allowed_file(file.filename):
#                 image = Image.open(file.stream)
#                 if image.mode == 'RGBA':
#                     image = image.convert('RGB')
#                 image = image.resize((1024, 768))
#
#                 result_image = process_image(image)
#                 imgByteArr = io.BytesIO()
#                 result_image.save(imgByteArr, format='JPEG')
#
#                 # imgByteArr = io.BytesIO()
#                 # image.save(imgByteArr, format='JPEG')
#                 image_data = imgByteArr.getvalue()
#
#                 new_image = Request_image(image_data=image_data, llm_answer_id=request_id)
#                 db.session.add(new_image)
#
#         db.session.commit()
#
#         return redirect(url_for('Request.create_assignq'))
#
#     return render_template('create_assignq.html', courses=courses)

@RequestBP.route('/Request', methods=['GET', 'POST'])
def create_assignq():
    courses = Course.query.all()  # 获取所有课程
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'create':
            qtext = request.form['qtext']
            course_id = request.form.get('course_id')  # 从表单获取课程 ID
            llm_used = request.form['llm_used']
            score = int(request.form['score'])
            comment = request.form['comment']

            new_request = Request(
                qtext=qtext,
                course_id=course_id,  # 保存课程 ID
                llm_used=llm_used,
                score=score,
                comment=comment,
                llm_answer_id=None,
                new_score=None,
                explanation=None,
                user_id=None,
                course_number=None,
                course_name=None,
                course_category=None,
                request_type='create_assignq'
            )
            db.session.add(new_request)
            db.session.flush()

            request_id = new_request.request_id

            files = request.files.getlist('images')  # 获取所有上传的图片文件
            for file in files:
                if file and allowed_file(file.filename):
                    image = Image.open(file.stream)
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    image = image.resize((1024, 768))

                    result_image = process_image(image)
                    imgByteArr = io.BytesIO()
                    result_image.save(imgByteArr, format='JPEG')

                    image_data = imgByteArr.getvalue()

                    new_image = Request_image(image_data=image_data, llm_answer_id=request_id)
                    db.session.add(new_image)

            db.session.commit()

            return redirect(url_for('Request.create_assignq'))

        elif action == 'retry':
            # 重新处理上传的图片
            files = request.files.getlist('images')
            for file in files:
                if file and allowed_file(file.filename):
                    image = Image.open(file.stream)
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    image = image.resize((1024, 768))

                    result_image = process_image(image)
                    imgByteArr = io.BytesIO()
                    result_image.save(imgByteArr, format='JPEG')

                    image_data = imgByteArr.getvalue()

                    # 处理完图片后显示结果或做其他操作
                    # 这里可以选择保存到数据库，或者展示在页面上
                    # 示例：将处理后的图片存入临时目录，并将链接返回到前端
                    # save_processed_image(image_data)

            return redirect(url_for('Request.create_assignq'))

    return render_template('create_assignq.html', courses=courses)


@RequestBP.route('/run_image_processing', methods=['POST'])
def run_image_processing():
    files = request.files.getlist('images')
    if files:
        file = files[0]  # 这里只处理第一张图片
        if file and allowed_file(file.filename):
            image = Image.open(file.stream)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image = image.resize((1024, 768))

            result_image = process_image(image)
            imgByteArr = io.BytesIO()
            result_image.save(imgByteArr, format='JPEG')
            imgByteArr.seek(0)

            return send_file(imgByteArr, mimetype='image/jpeg')

    return 'No image uploaded', 400


@RequestBP.route('/image/<int:image_id>')
def serve_image(image_id):
    image_record = LLM_Answer_Image.query.filter_by(id=image_id).first()
    if image_record:
        return Response(image_record.image_data, mimetype='image/jpeg')
    else:
        return 'Image Not Found', 404


@RequestBP.route('/admin/change_score', methods=['GET', 'POST'])
def admin_change_score():
    if request.method == 'GET':
        # Fetch all change_score type requests and join with LLM_Answer table
        requests = db.session.query(Request, LLM_Answer.score.label('current_score')).join(LLM_Answer, Request.llm_answer_id == LLM_Answer.LLM_id).filter(Request.request_type == 'change_score').all()
        return render_template('admin_change_score.html', requests=requests)
    elif request.method == 'POST':
        request_id = request.form.get('request_id')
        # Fetch the specific request based on request_id
        req = Request.query.get(request_id)
        if req:
            llm_answer = LLM_Answer.query.get(req.llm_answer_id)
            if llm_answer:
                # Update the score in LLM_Answer as per the new score in the request
                llm_answer.score = req.new_score
                # Commit the score update
                db.session.commit()
                # Delete the processed request from the database
                db.session.delete(req)
                # Commit the deletion
                db.session.commit()
        return redirect(url_for('Request.admin_change_score'))


@RequestBP.route('/admin/create_assignq', methods=['GET', 'POST'])
def admin_create_assignq():
    if request.method == 'GET':
        requests_with_images = []
        requests = Request.query.filter_by(request_type='create_assignq').all()
        for req in requests:
            images = Request_image.query.filter_by(llm_answer_id=req.request_id).all()
            encoded_images = [base64.b64encode(img.image_data).decode('utf-8') for img in images]
            requests_with_images.append((req, encoded_images))

        return render_template('admin_create_assignq.html', requests_with_images=requests_with_images)
    elif request.method == 'POST':
        request_id = request.form.get('request_id')
        req = Request.query.get(request_id)
        if req:
            llm_answer = LLM_Answer(LLM_used=req.llm_used, score=req.score, comment=req.comment)
            db.session.add(llm_answer)
            db.session.flush()

            assignq = AssignQ(Qtext=req.qtext, course_id=req.course_id, llm_answer=llm_answer)  # 保存课程 ID
            db.session.add(assignq)

            images = Request_image.query.filter_by(llm_answer_id=req.request_id).all()
            for image in images:
                new_image = LLM_Answer_Image(image_data=image.image_data, llm_answer_id=llm_answer.LLM_id)
                db.session.add(new_image)
                db.session.delete(image)

            db.session.delete(req)
            db.session.commit()

        return redirect(url_for('Request.admin_create_assignq'))


@RequestBP.route('/view_personal_experiments')
def view_personal_experiments():
    experiments = PersonalExperiment.query.filter_by(user_id=session.get('user_id')).all()
    return render_template('view_experiments.html', experiments=experiments)


@RequestBP.route('/view_upload')
def view_upload():
    requests_with_images = []
    requests = Request.query.filter_by(request_type='create_assignq').all()
    for req in requests:
        images = Request_image.query.filter_by(llm_answer_id=req.request_id).all()
        encoded_images = [base64.b64encode(img.image_data).decode('utf-8') for img in images]
        requests_with_images.append((req, encoded_images))

    return render_template('view_upload.html', requests_with_images=requests_with_images)