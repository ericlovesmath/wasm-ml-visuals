import * as d3 from "d3";
import { LCInSampleError } from "algs";

const width = 600;
const height = 600;
const marginTop = 20;
const marginRight = 20;
const marginBottom = 30;
const marginLeft = 40;

interface DataPoint {
  x: number;
  mean: number;
  std: number;
}

const x = d3
  .scaleLinear()
  .domain([10, 1000])
  .range([marginLeft, width - marginRight]);

const y = d3
  .scaleLinear()
  .domain([0, 0.05])
  .range([height - marginBottom, marginTop]);

const svg = d3
  .select("#inSampleErrorPlot")
  .attr("width", width)
  .attr("height", height);

// Mask out plot overflowing into axis
svg
  .append("defs")
  .append("SVG:clipPath")
  .attr("id", "clip")
  .append("SVG:rect")
  .attr("x", marginLeft)
  .attr("y", marginTop)
  .attr("width", width - marginLeft - marginRight)
  .attr("height", height - marginBottom - marginTop);

const mean = d3
  .line<DataPoint>()
  .x((d) => x(d.x))
  .y((d) => y(d.mean));

const std = d3
  .area<DataPoint>()
  .x((d) => x(d.x))
  .y0((d) => y(d.mean - d.std)) // Bottom edge of the area
  .y1((d) => y(d.mean + d.std)); // Top edge of the area

function plot_lc_in_sample_error() {
  // Plot x and y axis
  svg
    .append("g")
    .attr("transform", `translate(0, ${height - marginBottom})`)
    .call(d3.axisBottom(x));
  svg
    .append("g")
    .attr("transform", `translate(${marginLeft}, 0)`)
    .call(d3.axisLeft(y));

  // Simulation
  let data: DataPoint[] = [];
  let runner = LCInSampleError.new();
  for (let n = 10; n <= 1000; n += 10) {
    setTimeout(() => {
      runner.run(n);
      data.push({ x: n, mean: runner.mean, std: runner.std });

      svg.selectAll("path").remove();

      svg
        .datum(data)
        .append("path")
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 2)
        .attr("clip-path", "url(#clip)")
        .attr("d", mean);

      svg
        .datum(data)
        .append("path")
        .attr("fill", "lightblue")
        .attr("opacity", 0.5)
        .attr("clip-path", "url(#clip)")
        .attr("d", std);
    }, 0);
  }

  setTimeout(() => runner.free(), 0);
}

plot_lc_in_sample_error();
