const child_process = require('child_process');

var affine_forward = function(x, w, b) {
  return w * x + b;
}

var loss_function = function(z, y, callback) {
  return 0.5 * Math.pow(y - z, 2);
}

var loss_derivative = function(z, y, callback) {
  return z - y;
}

var xs = [];
var ys = [];

for (var i = -500; i < 500; i++) {
  xs.push(i / 100);
}
for (var i = -500; i < 500; i++) {
  ys.push(0.3 * (i / 100) + 0.2 + 0.3 * Math.random());
}

omega = 5 * Math.random();
beta = 5 * Math.random();

num_epoch = 200;
learning_rate = 1e-1;

console.log("Omega: " + omega);
console.log("Beta: " + beta);

for (var i = 0; i < num_epoch; i++) {
  var epoch_loss = 0;
  var omega_delta = 0;
  var beta_delta = 0;
  for (var k = 0; k < xs.length; k++) {
    var x = xs[k];
    var y = ys[k];
    estimation = affine_forward(x, omega, beta);
    epoch_loss += loss_function(estimation, y);
    dx = loss_derivative(estimation, y);
    omega_delta += dx * x;
    beta_delta += dx;
  }
  omega += -learning_rate * (omega_delta / xs.length)
  beta += -learning_rate * (beta_delta / xs.length)
  var loss = epoch_loss / xs.length;
  child_process.execSync("sleep 0.15")
  console.log("Loss: " + loss + ", Omega: " + omega + ", Beta: " + beta);
}
