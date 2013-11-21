// set the scene size
var WIDTH = 940, HEIGHT = 500;

// global variables
var viewer, camera, scene, renderer, ambient, sphere, controls;

// set some camera attributes
var VIEW_ANGLE = 70, ASPECT = WIDTH / HEIGHT,
	NEAR = 1, FAR = 10000;

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

	// load model
	// SphereGeometry: radius, segmentsWidth, segmentsHeight
	sphere = new THREE.Mesh(new THREE.SphereGeometry(150,100,100), new THREE.MeshNormalMaterial());
	// sphere.overdraw = true;
	scene.add(sphere);

	console.log("done initScene");
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

function load(filename) {
	geometry = new THREE.Geometry();
	geometry.dynamic = true;

	// get the point cloud
	// points have to be stored in a text file, with 1 point on each line: x y z r g b, separated by whitespace
	// ajax stuff

	// process file

}