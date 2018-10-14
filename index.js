let xData = [];
let yData = [];
let button, learningRateSlider;

// mx + b
let m, b;

let optimizer;

function setup() {
  createCanvas(750, 750);
  background(50);

  m = tf.variable(tf.scalar(random()));
  b = tf.variable(tf.scalar(random()));

  exportButton = createButton("Export data");
  exportButton.position(10, width + 20);
  exportButton.mousePressed(() =>
    console.log(JSON.stringify({ xData, yData }))
  );

  resetButton = createButton("Reset data");
  resetButton.position(240, width + 20);
  resetButton.mousePressed(() => {
    xData = [];
    yData = [];
  });

  retrainButton = createButton("Retrain");
  retrainButton.position(340, width + 20);
  retrainButton.mousePressed(() => {
    m.assign(tf.scalar(random()));
    b.assign(tf.scalar(random()));
  });

  learningRateSlider = createSlider(0, 0.5, 0.05, 0.01);
  learningRateSlider.position(100, width + 20);
}

function predict(xData) {
  let x = tf.tensor1d(xData);

  return x.mul(m).add(b);
}

function loss(yh, yData) {
  let y = tf.tensor1d(yData);

  return yh
    .sub(y)
    .square()
    .mean();
}

function mapData(data) {
  return map(data, 0, width, -1, 1);
}

function unmapData(data) {
  return map(data, -1, 1, 0, width);
}

function mouseClicked() {
  if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > width) return;

  xData.push(mapData(mouseX));
  yData.push(-mapData(mouseY));
}

function draw() {
  background(50);

  xData.forEach((x, i) => {
    fill(255);
    noStroke();
    ellipse(unmapData(x), width - unmapData(yData[i]), 8);
  });

  const lineX1 = mapData(width * 0.1);
  const lineX2 = mapData(width * 0.9);
  tf.tidy(() => {
    const [lineY1, lineY2] = predict([lineX1, lineX2]).dataSync();

    stroke(255, 125, 0);
    strokeWeight(3);
    line(
      unmapData(lineX1),
      width - unmapData(lineY1),
      unmapData(lineX2),
      width - unmapData(lineY2)
    );
  });

  noStroke();
  fill(255);
  if (xData.length > 1) {
    tf.tidy(() => {
      optimizer = tf.train.sgd(learningRateSlider.value());

      optimizer.minimize(() => loss(predict(xData), yData));
      // optimizeManualy();
      text(
        `Learning rate: ${learningRateSlider.value()} Loss: ${
          loss(predict(xData), yData).dataSync()[0]
        }`,
        10,
        height - 10
      );
    });
  } else text("Learning rate: " + learningRateSlider.value(), 10, height - 10);
}

function optimizeManualy() {
  const gradM = tf
    .tensor1d(yData)
    .sub(predict(xData))
    .mul(tf.tensor(xData))
    .mul(tf.scalar(-1))
    .mean()
    .mul(2);

  const gradB = tf
    .tensor1d(yData)
    .sub(predict(xData))
    .mul(tf.scalar(-1))
    .mean()
    .mul(2);

  m.assign(m.sub(tf.scalar(learningRate).mul(gradM)));
  b.assign(b.sub(tf.scalar(learningRate).mul(gradB)));
}
