import init from 'algs';

init().then(() => {
  import("./main.ts")
    .catch(e => console.error("Error importing `main.ts`:", e));
})
