// create the particle variables
var particleCount = 1000, // should be the no. of points in the point cloud
    particles = new THREE.Geometry(),
    pMaterial = new THREE.ParticleBasicMaterial({
        color: 0xFFFFFF,
        size: 1
    });

// create the individual particles
for (var p = 0; p < particleCount; p++) {
    // create a particle with its corresponding x, y and z-coordinates
    var pX = Math.random() * 500 - 250, // x-coord
        pY = Math.random() * 500 - 250, // y-coord
        pZ = Math.random() * 500 - 250, // z-coord
        particle = new THREE.Vertex( new THREE.Vector3(pX, pY, pZ) );

    particles.vertices.push(particle);
}

// create the particle system
var particleSystem = new THREE.ParticleSystem(particles, pMaterial);

// add the particle system to the scene
scene.addChild(particleSystem);

var doc = document.documentElement;
doc.ondragover = function () {
    this.className = 'hover';
    return false;
};
doc.ondragend = function () {
    this.className = '';
    return false;
};
doc.ondrop = function (event) {
    event.preventDefault && event.preventDefault;
    this.className = '';
    var files = event.dataTransfer.files;
    return false;
};

var formData = new FormData();
for (var i=0; i < files.length; i++) {
    formData.append('file', files[i]);
}

// post a new XHR request
var xhr = new XMLHttpRequest();
xhr.open('POST', '/upload');
xhr.onload = function () {
    if (xhr.status === 200) {
        console.log('finished uploading: ' + xhr.status);
    } else {
        console.log('an error occurred in uploading');
    }
};
xhr.send(formData);

var acceptedFileFormats = {
    'image/png': true,
    'image/jpeg': true,
    'image/jpg': true
};

if (acceptedTypes[file.type] === true) {
    var reader = new FileReader();
    reader.onload = function (event) {
        var image = new Image();
        image.src = event.target.result;
        document.body.appendChild(image);
    };
    reader.readAsDataURL(file);
}