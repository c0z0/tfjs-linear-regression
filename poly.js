let xData = [];
let yData = [];
let button, learningRateSlider;

// m1x^2 + m2x + b

let params = Array(2).fill(0);

let optimizer;

function setup() {
  createCanvas(750, 750);
  background(50);

  params = params.map(() => tf.variable(tf.scalar(random())));

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

  polyGrade = createSelect();
  polyGrade.position(340, width + 20);

  for (let i = 1; i < 15; i++) polyGrade.option(`${i} grade polynome`);
  polyGrade.changed(() => {
    const grade = parseInt(polyGrade.value().slice(0, 2));

    params = Array(grade + 1)
      .fill(0)
      .map(() => tf.variable(tf.scalar(random())));
  });

  retrainButton = createButton("Retrain");
  retrainButton.position(500, width + 20);
  retrainButton.mousePressed(() => {
    params.forEach(p => p.assign(tf.scalar(random())));
  });

  learningRateSlider = createSlider(0, 0.5, 0.1, 0.01);
  learningRateSlider.position(100, width + 20);
}

function predict(xData) {
  let x = tf.tensor1d(xData);

  return tf.tidy(() => {
    let y = params[0];
    for (let i = 1; i < params.length; i++) {
      y = y.add(params[i].mul(x.pow(i)));
    }
    return y;
  });
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

  beginShape();

  stroke(255, 125, 0);
  strokeWeight(4);
  noFill();

  let linexs = Array(Math.floor(2 / 0.01 + 1))
    .fill(0)
    .map((_, i) => -1 + i * 0.01);

  tf.tidy(() => {
    let lineys = predict(linexs).dataSync();
    linexs.forEach((x, i) => {
      vertex(unmapData(x), width - unmapData(lineys[i]));
    });
  });

  endShape();

  noStroke();
  fill(255);
  if (xData.length > 1) {
    tf.tidy(() => {
      optimizer = tf.train.sgd(learningRateSlider.value());

      optimizer.minimize(() => loss(predict(xData), yData));
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
