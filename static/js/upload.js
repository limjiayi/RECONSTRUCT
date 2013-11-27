var dropbox = document.getElementsByClassName('dropbox')[0];
if (dropbox !== undefined) {
    dropbox.addEventListener('dragenter', dragenter, false);
    dropbox.addEventListener('dragover', dragover, false);
    dropbox.addEventListener('drop', drop, false);
}

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
    startButton = document.getElementsByClassName('startbtn');
    clearButton = document.getElementsByClassName('clearbtn');
    field = document.getElementsByTagName('input');
    if (field.length === 0) {
        $field = $('<form><input id="field" type="text" name="cloud_name" placeholder="Type a name for your point cloud."></form>');
        $('#preview').append($field);
    }
    if (startButton.length === 0) {
        $startButton = $('<button class="startbtn on">Start!</button>');
        $('#preview').append($startButton);
    }
    if (clearButton.length === 0) {
        $clearButton = $('<button class="clearbtn on">Clear</button><div id="empty"></div>');
        $('#preview').append($clearButton);
    }
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var imageType = /image.*/;
        var $img = $('<img class="preview" src="" width="200" />');
        $('#preview').append($img);

        $img.click( function() {
            $(this).toggleClass('checked');
        });
        if (!file.type.match(imageType)) {
            alert("File is not an image!", imageType);
        }
        // create thumbnails of the selected photos
        // photos are stored in memory?
        var reader = new FileReader();
        reader.onload = (function(img) {
            return function(e) {
                img.src = e.target.result;
            };
        })($img.get(0));
        reader.readAsDataURL(file);
    }
}

function startEnable() {
    $(document).on("click", ".startbtn", function() {
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
                sendFiles(selectedPhotos);
            }
        }
   });
}

function clearPreview() {
    $(document).on("click", ".clearbtn", function() {
        previews = document.getElementsByClassName('preview');
        if (previews.length > 0) {
            $('.preview').remove();
        }
    });
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
        success: function(data) {
            console.log('Successfully uploaded files.');
            clear_scene();
            load_cloud(data);
            alert('Point cloud loaded!');
            pastClouds();
        }
    });
}

function chooseCloud(cloud_id) {
    $.ajax( {
        url: '/cloud/' + cloud_id,
        type: 'GET',
        async: true,
        dataType: 'text',
        success: function(data) {
            if (data === null) {
                alert('Loading of point cloud failed.');
            } else {
                clear_scene();
                load_cloud(data);
            }
        }
    });
}

function pastClouds() {
    $(document).ready(function() {
        var clouds_div = document.getElementById('clouds');
        if (clouds_div !== null) {
            user_id = $('#clouds').attr('data-user-id');
            $.ajax( {
                url: '/past/' + user_id,
                type: 'GET',
                async: true,
                dataType: 'json',
                success: function(data) {
                    var clouds = document.getElementsByClassName('cloud');
                    if (clouds.length > 0) {
                        $('.cloud').remove();
                    }
                    //display the previous point clouds if there are any
                    for (var key in data) {
                        if (data[key][0] && data[key][0]['cloud_id']) {
                            var $div = $('<div class="cloud" data-cloud-id="' + data[key][0]['cloud_id'] + '"><div class="desc">Name: ' + key + '<br>Created on: ' + data[key][0]['uploaded_on'].split(' ')[0] + '</div>'); // name of cloud
                            $('#clouds').append($div);

                            for (var i=0; i < data[key].length; i++) {
                                var $photo = $('<img class="thumbnail" src="' + data[key][i]['path'] + '/' + data[key][i]['filename'] + '">');
                                $div.append($photo);
                            }
                        }
                    }
                    $('.cloud').mouseenter( function() { // bind event listener
                        $(this).children('img').addClass('hover');
                        // add load button
                        var loadButtons = document.getElementsByClassName('loadbtn');
                        if (loadButtons.length === 0) {
                            $loadbtn = $('<button class="loadbtn">Load</button>');
                            $(this).append($loadbtn);
                            $loadbtn.toggleClass('on');
                            $loadbtn.bind('click', function() {
                                var cloud_id = $(this).parent('.cloud').data('cloud-id');
                                chooseCloud(cloud_id);
                            });
                        } // add delete button
                        var deleteButtons = document.getElementsByClassName('deletebtn');
                        if (deleteButtons.length ===0) {
                            $deletebtn = $('<button class="deletebtn">Delete</button>');
                            $(this).append($deletebtn);
                            $deletebtn.toggleClass('on');
                            $deletebtn.bind('click', function() {
                                var cloud_id = $(this).parent('.cloud').data('cloud-id');
                                $.ajax( {
                                    url: '/remove/' + user_id + '/' + cloud_id,
                                    type: 'POST',
                                    async: true,
                                    dataType: 'text',
                                    success: function(data) {
                                        console.log('Successfully deleted cloud.');
                                        pastClouds();
                                    }
                                });
                            });
                        }
                    });
                    $('.cloud').mouseleave( function() {
                        $(this).children('img').removeClass('hover');
                        var loadbuttons = document.getElementsByClassName('loadbtn');
                        if (loadbuttons.length > 0) {
                            $('.loadbtn').remove();
                        }
                        var deletebuttons = document.getElementsByClassName('deletebtn');
                        if (deletebuttons.length > 0) {
                            $('.deletebtn').remove();
                        }
                    });
                }
            });
        }
    });
}


jQuery(function() {
  startEnable();
  clearPreview();
  pastClouds();
});
