var numRows = 0;
function createStatusbar(obj) {
    numRows++;
    var row = 'odd';
    if (numRows % 2 === 0) {row = 'even';}
    this.statusbar = $("<div class = 'statusBar "+row+"'></div>");
    this.filename = $("<div class = 'filename'></div>").appendTo(this.statusbar);
    this.progressBar = $("<div class = 'progressBar'><div></div></div>").appendTo(this.statusbar);
    this.abort = $("<div class='abort'>Abort</div>").appendTo(this.statusbar);
    obj.after(this.statusbar);

    this.setFileNameSize = function (name, size) {
        var sizeStr = '';
        var sizeKB = size / 1024;
        if (parseInt(sizeKB, 10) > 1024) {
            var sizeMB = sizeKB / 1024;
            sizeStr = sizeMB.toFixed(2) + ' MB';
        } else {
            sizeStr = sizeKB.toFixed(2) + 'KB';
        }
        this.filename.html(name);
        this.size.html(sizeStr);
    };
    this.setProgress = function (progress) {
        var progressBarWidth = progress * this.progressBar.width() / 100;
        this.progressBar.find('div').animate({width: progressBarWidth}, 10).html(progress + '% ');
        if (parseInt(progress, 10) >= 100) {
            this.abort.hide();
        }
    };
    this.setAbort = function (jqxhr) {
        var sb = this.statusbar;
        this.abort.click(function () {
            jqxhr.abort();
            sb.hide();
        });
    };
}

// read the file contents using HTML5 FormData() when the files are dropped
function handleFileUpload(files, obj) {
    var formData = new FormData();
    for (var i=0; i < files.length; i++) {
        formData.append('file', files[i]);

        var status = new createstatusbar(obj);
        status.setFileNameSize(files[i].name, files[i].size);
        sendFileToServer(formData, status);
    }
}

// send FormData() to the server using jQuery's AJAX API
function sendFileToServer(formData, status) {
    var jqXHR = $.ajax({
        xhr: function () {
            var xhrObj = $.ajaxSettings.xhr();
            if (xhrObj.upload) {
                xhrObj.upload.addEventListener('progress', function (event) {
                    var percent = 0;
                    var position = event.loaded || event.position;
                    var total = event.total;
                    if (event.lengthComputable) {
                        percent = Math.ceil(position / total) * 100;
                    }
                    // set progress
                    status.setProgress(percent);
                }, false);
            }
            return xhrObj;
        },
        type: 'POST',
        contentType: false,
        processData: false,
        cache: false,
        data: JSON.stringify(formData),
        success: function (data) {
            status.setProgress(100);
            $('#uploadStatus').append("File upload done <br>");
        }
    });
    status.setAbort(jqXHR);
}

$(document).ready(function () {
    // drag-n-drop functionality for uploading photos using jQuery
    var obj = $('.dragAndDrop_off');
    obj.on('dragenter', function (e) {
        e.stopPropagation();
        e.preventDefault();
        var box = document.getElementById('dragAndDropHandler');
        box.className = 'dragAndDrop_on';
    });

    obj.on('dragover', function (e) {
        e.stopPropagation();
        e.preventDefault();
        var box = document.getElementById('dragAndDropHandler');
        box.className = 'dragAndDrop_off';
    });

    obj.on('drop', function (e) {
        e.preventDefault();
        var files = e.originalEvent.dataTransfer.files;
        handleFileUpload(files, obj);
    });

    // prevent files from being opened in the browser window when files are dropped outside of the div
    $(document).on('dragenter', function (e) {
        e.stopPropagation();
        e.preventDefault();
    });

    $(document).on('dragover', function (e) {
        e.stopPropagation();
        e.preventDefault();
    });

    $(document).on('dragover', function (e) {
        e.stopPropagation();
        e.preventDefault();
    });
});