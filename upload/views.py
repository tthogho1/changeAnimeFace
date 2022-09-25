from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from upload.models import Document
from upload.forms import DocumentForm
from upload.classes.faceChange import FaceChange
import base64

def index(request):
    documents = Document.objects.all()
    return render(request, 'index.html', { 'documents': documents })

def basic_upload(request):
    if request.method == 'POST' and request.FILES['testfile']:
        myfile = request.FILES['testfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        faceChange = FaceChange()
        convert_file_url = faceChange.changeAnimeFace(uploaded_file_url)
        return render(request, 'basic_upload.html', {
            'uploaded_file_url': convert_file_url
        })
    return render(request, 'basic_upload.html')

def camera(request):
    if request.method == 'POST' and request.POST['faceImg']:
        faceImg = request.POST['faceImg']
        faceImg = faceImg.split(',')[1] # remove data:image/jpeg;base64,
        faceBynary = base64.b64decode(faceImg)

        f = open('C:/SourceCode/changeAnimeFace/media/myfile.jpg', 'wb')
        f.write(faceBynary)
        f.close()

        uploaded_file_url = '/media/myfile.jpg'
        faceChange = FaceChange()
        convert_file_url = faceChange.changeAnimeFace(uploaded_file_url)
        return render(request, 'camera.html', {
            'uploaded_file_url': convert_file_url
        })
    return render(request, 'camera.html')

def modelform_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = DocumentForm()
    return render(request, 'modelform_upload.html', {
        'form': form
    })