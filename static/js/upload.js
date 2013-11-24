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
    var files = e.dataTransfer.files;
    handleFiles(files);
}

function handleFiles(files) {
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var imageType = /image.*/;
        var $img = $('<img src="" width="200" />');
        $('#preview').append($img);

        $img.click( function() {
            $(this).toggleClass('checked');
        });

        if (!file.type.match(imageType)) {
            console.log("File is not an image!", imageType);
        }
     
        // create thumbnails of the selected photos
        // photos are stored on disk
        var reader = new FileReader();
        reader.onload = (function(img) {
            return function(e) {
                img.src = e.target.result;
            };
        })($img.get(0));
        reader.readAsDataURL(file);
    }
    button = document.getElementsByClassName('startbtn');
    field = document.getElementsByTagName('input');
    if (field.length === 0) {
        $field = $('<form><input id="field" type="text" name="cloud_name" placeholder="Type a name for your point cloud."></form>');
        $('#preview').append($field);
    }
    if (button.length === 0) {
        $button = $('<button class="startbtn">Start!</button>');
        $('#preview').append($button);
    }
    // startEnable();
}

function startEnable() {
  $(document).on("click", ".startbtn", function() {
        $('.startbtn').addClass('on');
        console.log('clicked start');
        var selectedPhotos = [];
        var photos = document.getElementsByClassName('checked');
        if (photos.length < 2) {
            alert('Minimum of 2 photos required.');
        } else {
            var input = document.getElementById('field').value;
            if (input === '') {
                alert('Please input a name for your point cloud.');
            } else {
                for (var i=0; i < photos.length; i++) {
                selectedPhotos.push(photos[i].src);
                }
                console.log('# photos: ', photos.length);
                sendFiles(selectedPhotos);
                startDisable();
            }
        }
   });
}

function startDisable() {
    $('img').unbind('click');
    $('.startbtn').unbind('click');
    $('.startbtn').removeClass('on');
}

function sendFiles(photos) {
    formData = new FormData();
    for (var i=0; i < photos.length; i++) {
        formData.append('photo['+i+']', photos[i]);
    }
    var input = document.getElementById('field').value;
    formData.append('cloud_name', input);
    uploadFiles(formData);
}

function uploadFiles(formData) {
    var uploadURL ="/upload";
    var jqXHR=$.ajax( {
        xhr: function() {
            var xhrobj = $.ajaxSettings.xhr();
            if (xhrobj.upload) {
                xhrobj.upload.addEventListener('progress', function(event) {
                    var percent = 0;
                    var position = event.loaded || event.position;
                    var total = event.total;
                    if (event.lengthComputable) {
                        percent = Math.ceil(position / total * 100);
                    }
                }, false);
            }
            return xhrobj;
        },
        url: uploadURL,
        type: "POST",
        contentType:false,
        processData: false,
        cache: false,
        data: formData,
        success: function(data){
            console.log('Successfully uploaded files.');
            load_cloud(data);
        }
    });
}

function chooseCloud() {
    $('.cloud a').bind('click', function() {
        $(this).addClass('selected');
        cloud_id = $(this).attr('data-cloud-id');

        $.ajax( {
            url: '/cloud/' + cloud_id,
            type: 'GET',
            async: true,
            dataType: 'text',
            success: function(data) {
                if (data === null) {
                    alert('Loading of point cloud failed.');
                } else {
                    load_cloud(data);
                }
            }
        });
    });
}

function pastClouds() {
    var cloud_div = document.getElementById('cloud');
    user_id = getAttribute('data-user-id');

    $.ajax( {
        url: '/display_user/' + user_id,
        type: 'GET',
        async: true,
        dataType: 'text',
        success: function(data) {
            if (data === null) {
                alert('Loading of point cloud failed.');
            } else {
                load_cloud(data);
            }
        }
    });
}

jQuery(function(){
  startEnable();
});
