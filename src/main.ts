import "./style.css";
import Chart from "chart.js/auto";
import { wasm_memory, LinearClassifier } from "algs";

const canvas = <HTMLCanvasElement>document.getElementById("myChart");

let chart = new Chart(canvas, {
  type: "line",
  data: {
    labels: [] as Number[],
    datasets: [
      {
        label: "In Sample Error vs N, Averaged over 300",
        backgroundColor: "rgba(0,0,255,1.0)",
        borderColor: "rgba(0,0,255,0.1)",
        data: [] as Number[],
      },
      {
        label: "+1 STD",
        type: "line",
        backgroundColor: "rgb(75, 192, 255, 0.5)",
        borderColor: "transparent",
        pointRadius: 0,
        fill: 0,
        tension: 0.5,
        data: [] as Number[],
        yAxisID: "y",
        xAxisID: "x",
      },
      {
        label: "-1 STD",
        type: "line",
        backgroundColor: "rgb(75, 192, 255, 0.5)",
        borderColor: "transparent",
        pointRadius: 0,
        fill: 0,
        tension: 0.5,
        data: [] as Number[],
        yAxisID: "y",
        xAxisID: "x",
      },
    ],
  },
  options: {
    devicePixelRatio: 2,
    scales: { y: { min: 0, max: 0.05 } },
  },
});

const random_sample = (n: number, m: number, b: number) => {
  let xs = [...Array(n).keys()].map(
    (_) => Math.random() * 2 - 1,
    Math.random() * 2 - 1
  );
  let ys = [...Array(n).keys()].map(
    (_) => Math.random() * 2 - 1,
    Math.random() * 2 - 1
  );
  let target = [...Array(n).keys()].map(
    (i) => ys[i] > (xs[i] * m + b) ? 1 : -1
  );
  return [xs, ys, target];
};

let lc = LinearClassifier.new();
for (let n = 10; n <= 1000; n += 10) {
  setTimeout(() => {
    chart.data.labels!.push(n);
    let runs = 300;

    let error_ins = [...Array(runs).keys()].map((_) => {
      let [m, b] = [-1, 0.3];
      let [xs, ys, target] = random_sample(n, m, b);
      lc.train(xs, ys, target);
      let prediction = new Float64Array(wasm_memory().buffer, lc.predict(xs, ys), n);
      let diff = 0
      for (let i = 0; i < n; i += 1) {
        if (prediction[i] != target[i]) {
          diff += 1
        }
      }
      return diff / n;
    });

    let mean = error_ins.reduce((a, b) => a + b, 0) / runs;
    let std = Math.sqrt(
      error_ins.reduce((a, b) => a + (b - mean) * (b - mean), 0) / runs
    );

    chart.data.datasets[0].data.push(mean);
    chart.data.datasets[1].data.push(mean - std);
    chart.data.datasets[2].data.push(mean + std);
    chart.update("none");
  }, 0);
}

let scatter = new Chart("myScatter", {
  type: "scatter",
  data: {
    datasets: [
      {
        label: "Red",
        backgroundColor: "Red",
        data: [] as any[],
      },
      {
        label: "Blue",
        backgroundColor: "Blue",
        data: [] as any[],
      },
      {
        type: "line",
        label: "none",
        data: [] as any[],
      },
    ],
  },
  options: {
    devicePixelRatio: 2,
    scales: {
      x: { min: -1, max: 1 },
      y: { min: -1, max: 1 },
    },
    plugins: { legend: { labels: { filter: (item) => item.text !== "none" } } },
  },
});

// Accessing WASM Memory
let [n, m, b] = [25, Math.random() * 8 - 4, Math.random() * 1.6 - 0.8];
let [xs, ys, target] = random_sample(n, m, b);
scatter.data.datasets[2].data.push({ x: -2, y: -2 * m + b });
scatter.data.datasets[2].data.push({ x: 2, y: 2 * m + b });

lc.train(xs, ys, target);

for (let i = 0; i < n; i += 1) {
  scatter.data.datasets[target[i] == 1.0 ? 0 : 1].data.push({
    x: xs[i],
    y: ys[i],
  });
}
scatter.update("none");
