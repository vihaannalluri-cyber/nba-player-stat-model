const form = document.querySelector("#prediction-form");
const error = document.querySelector("#error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";
  const button = form.querySelector("button");
  button.disabled = true;
  button.firstChild.textContent = "Calculating… ";

  const payload = {
    player: document.querySelector("#player").value,
    opponent: document.querySelector("#opponent").value,
    date: document.querySelector("#game-date").value,
    location: document.querySelector('input[name="location"]:checked').value,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");

    document.querySelector("#pts").textContent = result.prediction.PTS.toFixed(1);
    document.querySelector("#reb").textContent = result.prediction.REB.toFixed(1);
    document.querySelector("#ast").textContent = result.prediction.AST.toFixed(1);
    document.querySelector("#result-context").textContent =
      `${result.player} · ${result.location === "home" ? "vs" : "at"} ${result.opponent}`;
  } catch (failure) {
    error.textContent = failure.message;
  } finally {
    button.disabled = false;
    button.firstChild.textContent = "Run projection ";
  }
});
