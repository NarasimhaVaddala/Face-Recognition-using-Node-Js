const faceapi = require('face-api.js');
const canvas = require('canvas');
const fs = require('fs');
const path = require('path');
const { Canvas, Image, ImageData } = canvas;

// Required for Node.js environment to work with face-api.js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function loadModels() {
  // Load models from the 'models' directory
  const modelPath = path.join(__dirname, 'models');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
}

async function loadImage(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
  const img = await canvas.loadImage(imageBuffer);
  return img;
}

async function compareFaces(image1Path, image2Path) {
  const img1 = await loadImage(image1Path);
  const img2 = await loadImage(image2Path);

  // Detect faces and extract face descriptors
  const face1 = await faceapi.detectSingleFace(img1).withFaceLandmarks().withFaceDescriptor();
  const face2 = await faceapi.detectSingleFace(img2).withFaceLandmarks().withFaceDescriptor();

  if (!face1 || !face2) {
    console.log('One or both images do not contain faces.');
    return;
  }

  // Compare the two face descriptors
  const distance = faceapi.euclideanDistance(face1.descriptor, face2.descriptor);

  if (distance < 0.5) { // Threshold of 0.6 for a good match
    console.log('Faces match! Distance: ', distance);
  } else {
    console.log('Faces do not match. Distance: ', distance);
  }
}

(async () => {
  // Load models before running face matching
  await loadModels();

  // Compare two images
  // Give path to images which you need to compare
  const image1Path = './img7.jpg';
  const image2Path = './img2.jpg';
  
  await compareFaces(image1Path, image2Path);
})();





/*
These are the models which i have downloaded from https://github.com/justadudewhohacks/face-api.js/tree/master/weights
ssd_mobilenetv1_model-weights_manifest.json
    │   ├── ssd_mobilenetv1_model-shard1
    │   ├── ssd_mobilenetv1_model-shard2
    │   ├── face_landmark_68_model-weights_manifest.json
    │   ├── face_landmark_68_model-shard1
    │   ├── face_recognition_model-weights_manifest.json
    │   ├── face_recognition_model-shard1

*/ 