import "./style.css";
import Chart from "chart.js/auto";
import { wasm_memory, LinearClassifier } from "algs";

const InputNumPoints = <HTMLInputElement>document.getElementById("num-points");
const InputNumRuns = <HTMLInputElement>document.getElementById("num-runs");
const ButtonRunBiasSim = <HTMLButtonElement>(
  document.getElementById("run-bias-sim")
);

const nMin = parseInt(InputNumPoints.min);
const nMax = parseInt(InputNumPoints.max);
const runsMin = parseInt(InputNumRuns.min);
const runsMax = parseInt(InputNumRuns.max);

let chart = new Chart("myChart", {
  type: "line",
  data: {
    labels: [] as Number[],
    datasets: [
      {
        label: "In Sample Error vs N, Averaged over 250",
        backgroundColor: "rgba(0,0,255,1.0)",
        borderColor: "rgba(0,0,255,0.2)",
        data: [] as Number[],
      },
      {
        label: "+1 STD",
        type: "line",
        backgroundColor: "rgba(75, 192, 255, 0.5)",
        pointRadius: 0,
        fill: 2,
        tension: 0.5,
        showLine: false,
        data: [] as Number[],
      },
      {
        label: "-1 STD",
        type: "line",
        backgroundColor: "rgba(75, 192, 255, 0.5)",
        pointRadius: 0,
        fill: 1,
        tension: 0.5,
        showLine: false,
        data: [] as Number[],
      },
    ],
  },
  options: {
    devicePixelRatio: 2,
    scales: { y: { min: 0, max: 0.05 } },
    animation: false,
  },
});

let scatter = new Chart("bias-sim-scatter", {
  type: "scatter",
  data: {
    datasets: [
      {
        type: "line",
        borderColor: "Black",
        label: "Hypothesis",
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

function random_line(): [number, number] {
  let x_1 = Math.random() * 2 - 1;
  let y_1 = Math.random() * 2 - 1;
  let x_2 = Math.random() * 2 - 1;
  let y_2 = Math.random() * 2 - 1;

  let m = (y_2 - y_1) / (x_2 - x_1);
  let b = y_1 - m * x_1;

  return [m, b];
}

function random_sample(n: number, hyp: (x: number) => number) {
  let xs = [...Array(n).keys()].map(
    (_) => Math.random() * 2 - 1,
    Math.random() * 2 - 1
  );
  let ys = [...Array(n).keys()].map(
    (_) => Math.random() * 2 - 1,
    Math.random() * 2 - 1
  );
  let labels = [...Array(n).keys()].map((i) => (ys[i] > hyp(xs[i]) ? 1 : -1));
  let sample = [...xs, ...ys];
  return [sample, labels];
}

function plot_lc_in_sample_error() {
  let lc = LinearClassifier.new();
  let [m, b, runs] = [-1, 0.3, 250];
  for (let n = 10; n <= 1000; n += 10) {
    setTimeout(() => {
      let error_ins = [...Array(runs).keys()].map((_) => {
        let [sample, labels] = random_sample(n, (x) => m * x + b);
        lc.train(n, sample, labels);
        let pred = new Float64Array(wasm_memory().buffer, lc.predict(n, sample), n);
        let diff = 0;
        for (let i = 0; i < n; i += 1) {
          if (pred[i] != labels[i]) {
            diff += 1;
          }
        }
        return diff / n;
      });

      let mean = error_ins.reduce((a, b) => a + b, 0) / runs;
      let std = Math.sqrt(
        error_ins.reduce((a, b) => a + (b - mean) * (b - mean), 0) / runs
      );

      chart.data.labels!.push(n);
      chart.data.datasets[0].data.push(mean);
      chart.data.datasets[1].data.push(mean - std);
      chart.data.datasets[2].data.push(mean + std);
      chart.update();
    }, 0);
  }
  setTimeout(() => lc.free(), 0);
}

function plot_lc_variance(n: number, runs: number) {
  let lc = LinearClassifier.new();
  let [m, b] = random_line();

  scatter.data.datasets = [
    {
      type: "line",
      borderColor: "Black",
      label: "Hypothesis",
      data: [
        { x: -2, y: -2 * m + b },
        { x: 2, y: 2 * m + b },
      ],
    },
  ];

  for (let i = 0; i < runs; i += 1) {
    let [sample, labels] = random_sample(n, (x) => m * x + b);
    lc.train(n, sample, labels);
    let w = new Float64Array(wasm_memory().buffer, lc.get_weights(), 3);
    scatter.data.datasets.push({
      type: "line",
      label: "none",
      borderColor: "rgba(100, 100, 100, 0.1)",
      data: [
        { x: -2, y: (-1 * (w[0] - 2 * w[1])) / w[2] },
        { x: 2, y: (-1 * (w[0] + 2 * w[1])) / w[2] },
      ],
    });
  }

  scatter.update("none");
  lc.free();
}

plot_lc_in_sample_error();
plot_lc_variance(InputNumPoints.valueAsNumber, InputNumRuns.valueAsNumber);

ButtonRunBiasSim.onclick = () => {
  let n = InputNumPoints.valueAsNumber;
  let runs = InputNumRuns.valueAsNumber;
  if (isNaN(n) || n < nMin || n > nMax) {
    alert(`Size of sample must be an integer from ${nMin} to ${nMax}`);
  } else if (isNaN(runs) || runs < runsMin || runs > runsMax) {
    alert(`Number of runs must be an integer from ${runsMin} to ${runsMax}`);
  } else {
    plot_lc_variance(n, runs);
  }
};
