import "./style.css";
import Chart from "chart.js/auto";
import { LCInSampleError, LCBiasVariance } from "algs";

const InputNumPoints = <HTMLInputElement>document.getElementById("num-points");
const InputNumRuns = <HTMLInputElement>document.getElementById("num-runs");
const ButtonRunBiasSim = <HTMLButtonElement>(
  document.getElementById("run-bias-sim")
);

const nMin = parseInt(InputNumPoints.min);
const nMax = parseInt(InputNumPoints.max);
const runsMin = parseInt(InputNumRuns.min);
const runsMax = parseInt(InputNumRuns.max);

let chart = new Chart("in-sample-error-canvas", {
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

let scatter = new Chart("bias-sim-canvas", {
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

function plot_lc_in_sample_error() {
  let runner = LCInSampleError.new();
  for (let n = 100; n <= 1000; n += 10) {
    setTimeout(() => {
      runner.run(n);

      chart.data.labels!.push(n);
      chart.data.datasets[0].data.push(runner.mean);
      chart.data.datasets[1].data.push(runner.mean - runner.std);
      chart.data.datasets[2].data.push(runner.mean + runner.std);
      chart.update();
    }, 0);
  }
  setTimeout(() => runner.free(), 0);
}

function plot_lc_variance(n: number, runs: number) {
  let runner = LCBiasVariance.new();
  let [m, b] = [runner.m, runner.b];

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
    runner.run(n, m, b);

    // Decision Boundary for [1, x, y]
    scatter.data.datasets.push({
      type: "line",
      label: "none",
      borderColor: "rgba(100, 100, 100, 0.1)",
      data: [
        { x: -2, y: runner.y_neg_two },
        { x: 2, y: runner.y_pos_two },
      ],
    });
  }

  scatter.update("none");
  runner.free();
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

function plot_lc_nonlinear() {
  let plot = new Chart("nonlinear-features-canvas", {
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
      plugins: {
        legend: { labels: { filter: (item) => item.text !== "none" } },
      },
    },
  });
}

plot_lc_nonlinear();
