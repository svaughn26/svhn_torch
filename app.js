console.log("JS file loaded");

async function loadModel() {
  const session = await ort.InferenceSession.create("svhn_cnn.onnx");
  return session;
}

async function runInference(session, imageData) {
  const tensor = new ort.Tensor("float32", imageData, [1, 3, 32, 32]);
  const results = await session.run({ input: tensor });
  const output = results.logits.data;
  const predictedDigit = output.indexOf(Math.max(...output));
  return predictedDigit;
}

document.getElementById("predictBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("imageInput");
  const resultText = document.getElementById("result");

  if (!fileInput.files.length) {
    resultText.textContent = "Please upload an image first.";
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(fileInput.files[0]);

  img.onload = async () => {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.drawImage(img, 0, 0, 32, 32);

    const imageData = ctx.getImageData(0, 0, 32, 32);
    const data = Float32Array.from(imageData.data).filter((_, i) => i % 4 !== 3);

    const session = await loadModel();
    const prediction = await runInference(session, data);

    resultText.textContent = "Prediction: " + prediction;
  };
});
