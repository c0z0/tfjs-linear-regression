let xData = [];
let yData = [];

// mx + b
let m, b;

let learningRate = 0.05;
let optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(750, 750);
  background(50);

  m = tf.variable(tf.scalar(random()));
  b = tf.variable(tf.scalar(random()));
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

  if (xData.length > 1) {
    tf.tidy(() => {
      noStroke();

      optimizer.minimize(() => loss(predict(xData), yData));
      // optimizeManualy();
      text(
        "Loss: " + loss(predict(xData), yData).dataSync()[0],
        10,
        height - 10
      );
    });
  }
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
