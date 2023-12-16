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
    scales: {
      y: {
        min: 0,
        max: 0.05,
      },
    },
  },
});

const random_sample = (n: number) => {
  let xs = [...Array(n).keys()].map(
    (_) => Math.random() * 2 - 1,
    Math.random() * 2 - 1
  );
  let ys = [...Array(n).keys()].map(
    (_) => Math.random() * 2 - 1,
    Math.random() * 2 - 1
  );
  return [xs, ys];
};

let lc = LinearClassifier.new();
for (let n = 10; n <= 1000; n += 10) {
  setTimeout(() => {
    chart.data.labels!.push(n);
    let runs = 300;

    let error_ins = [...Array(runs).keys()].map((_) => {
      let [xs, ys] = random_sample(n);
      lc.init(n, xs, ys);
      lc.train();
      return lc.in_sample_error();
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

// Accessing WASM Memory
let n = 10;
let [xs, ys] = random_sample(n);
lc.init(n, xs, ys);
lc.train();
let target = new Float64Array(wasm_memory().buffer, lc.get_target(), n);
console.log(target);
