// vulnerable.js
const { exec } = require("child_process");

function run(cmd) {
  // ❌ VULNERABLE
  exec(cmd, (err, stdout) => {
    console.log(stdout);
  });
}

run("ls " + userInput);
