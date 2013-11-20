// set the scene size
var WIDTH = 400, HEIGHT = 300;

// set some camera attributes
var VIEW_ANGLE = 45, ASPECT = WIDTH / HEIGHT,
	NEAR = 0.1, FAR = 10000;

function onLoad() {
	initScene();

	function initScene() {
		var viewer = document.getElementById('viewer');

		renderer = new THREE.WebGLRenderer( { antialias: true } );
		renderer.setSize( window.innerWidth, window.innerHeight );
		viewer.appendChild( renderer.domElement );

		// this is the scene all objects will be added to
		scene = new THREE.Scene();
		// ambient light
		var ambient = new THREE.AmbientLight( 0x999999 );
		scene.add( ambient );

		camera = new THREE.PerspectiveCamera(75, ASPECT, NEAR, FAR);
		camera.position.z = 10;
		scene.add( camera );

		// set the camera's behaviour and sensitivity
		controls = new THREE.TrackballControls( camera );
		controls.rotateSpeed = 5.0;
		controls.zoomSpeed = 5;
		controls.panSpeed = 2;
		controls.noZoom = false;
		controls.noPan = false;
		controls.staticMoving = true;
		controls.dynamicDampingFactor = 0.3;

		// load model
		// SphereGeometry: radius, segmentsWidth, segmentsHeight
		var sphere = new THREE.Mesh(new THREE.SphereGeometry(150,100,100), new THREE.MeshNormalMaterial());
		sphere.overdraw = true;
		scene.add(sphere);

		animate();
	}
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