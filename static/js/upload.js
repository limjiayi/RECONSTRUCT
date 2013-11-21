var dropbox = document.getElementsByClassName('dropbox')[0];
dropbox.addEventListener('dragenter', dragenter, false);
dropbox.addEventListener('dragover', dragover, false);
dropbox.addEventListener('drop', drop, false);

function dragenter(e) {
    e.stopPropagation();
    e.preventDefault();
    e.currentTarget.classList.add('on');
}

function dragover(e) {
    e.stopPropagation();
    e.preventDefault();
    e.currentTarget.classList.remove('on');
}

function drop(e) {
    e.stopPropagation();
    e.preventDefault();
    var dataTransfer = e.dataTransfer; // get the dataTransfer field from the event
    var files = dataTransfer.files;
    handleFiles(files);
}

function handleFiles(files) {
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var imageType = /image.*/;
    
        if (!file.type.match(imageType)) {
        continue;
        }
     
        var reader = new FileReader();
        reader.onload = (function(aImg) { return function(e) { aImg.src = e.target.result; }; })(img);
        reader.readAsDataURL(file);
    }
    sendFiles(files);
}

function sendFiles(files) {
    for (var i=0; i < files.length; i++) {
        new FileUpload(files[i], files[i].file);
    }
}

function FileUpload(img, file) {
    var reader = new FileReader();
    this.ctrl = createThrobber(img);
    var xhr = new XMLHttpRequest();
    this.xhr = xhr;

    var self = this;
    this.xhr.upload.addEventListener('progress', function(e) {
        if (e.lengthComputable) {
            var percentage = Math.round((e.loaded * 100) / e.total);
            this.ctrl.update(percentage);
        }
    }, false);

    xhr.upload.addEventListener('load', function(e) {
        this.ctrl.update(100);
        var canvas = self.ctrl.ctx.canvas;
        canvas.parentNode.removeChild(canvas);
    }, false);

    xhr.open('POST', 'localhost:5000');
    xhr.overrideMimeType('text/plain, charset=x-user-defined-binary');
    reader.onload = function(evt) {
        xhr.sendAsBinary(evt.target.result);
    };
    reader.readAsBinaryString(file);
}