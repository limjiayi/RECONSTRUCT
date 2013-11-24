// set the scene size
var WIDTH = 940, HEIGHT = 500;

// global variables
var viewer, camera, scene, renderer, ambient, sphere, controls;

// set some camera attributes
var VIEW_ANGLE = 70, ASPECT = WIDTH / HEIGHT,
	NEAR = 0.0001, FAR = 10000;

// set size of the particles
var particleSize = 0.001;

// keep track of the mouse position
var mouseX = 0, mouseY = 0;

function onLoad() {
	initScene();
	animate();
}

function initScene() {
	viewer = document.getElementById('viewer');

	renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.setSize( WIDTH, HEIGHT );
	viewer.appendChild( renderer.domElement );

	// this is the scene all objects will be added to
	scene = new THREE.Scene();
	// ambient light
	ambient = new THREE.AmbientLight( 0x999999 );
	scene.add( ambient );

	camera = new THREE.PerspectiveCamera(VIEW_ANGLE, ASPECT, NEAR, FAR);
	camera.position.z = 500;
	// scene.add( camera );

	// set the camera's behaviour and sensitivity
	controls = new THREE.TrackballControls( camera, viewer );
	controls.rotateSpeed = 5.0;
	controls.zoomSpeed = 5;
	controls.panSpeed = 2;
	controls.noZoom = false;
	controls.noPan = false;
	controls.staticMoving = true;
	controls.dynamicDampingFactor = 0.3;

	// SphereGeometry: radius, segmentsWidth, segmentsHeight
	// sphere = new THREE.Mesh(new THREE.SphereGeometry(150,100,100), new THREE.MeshNormalMaterial());
	// sphere.overdraw = true;
	// scene.add(sphere);
}

// the loop
function animate() {
	// this function calls itself on every frame
	requestAnimationFrame( animate );

	// on every frame, calculate the new camera position and have it look at the center of the scene
	controls.update();
	camera.lookAt(scene.position);
	renderer.render(scene, camera);
}

function load_cloud(data) {
	model = new THREE.Geometry();
	model.dynamic = true;

	// points are stored in a giant string, with 1 point on each line: x y z r g b, separated by whitespace
	var cloud = data.split('\n');
	console.log(cloud);

	// initialize min and max coords
	min_x = 0;
	min_y = 0;
	min_z = 0;
	max_x = 0;
	max_y = 0;
	max_z = 0;

	colours = [];

	// load the points
	for (var i=0; i<cloud.length; i++) {
		var pt = cloud[i].split(" ");
		var x = parseFloat(pt[0]);
		var y = parseFloat(pt[1]);
		var z = parseFloat(pt[2]);

		if (x < min_x) {min_x = x;}
		if (x > max_x) {max_x = x;}
		if (y < min_y) {min_y = y;}
		if (y > max_y) {max_y = y;}
		if (z < min_z) {min_z = z;}
		if (z > max_z) {max_z = z;}

		var colour = 'rgb(' + pt[3] + ',' + pt[4] + ',' + pt[5] + ')';
		model.vertices.push( new THREE.Vector3(x, y, z) );
		colours.push( new THREE.Color(colour) );
	}
	model.colors = colours;

	// load model
	var material = new THREE.ParticleBasicMaterial({ size: particleSize, vertexColors: true });
	particles = new THREE.ParticleSystem(model, material);
	scene.add(particles);
}