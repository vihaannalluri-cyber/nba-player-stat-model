const form = document.querySelector("#prediction-form");
const button = form.querySelector("button");
const errorMessage = document.querySelector("#error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = "Running model...";
  errorMessage.textContent = "";

  const data = {
    player: document.querySelector("#player").value,
    opponent: document.querySelector("#opponent").value,
    date: document.querySelector("#game-date").value,
    location: document.querySelector('input[name="location"]:checked').value,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error);
    }

    const location = result.location === "home" ? "vs." : "at";
    document.querySelector("#result-title").textContent = `${result.player} ${location} ${result.opponent}`;
    document.querySelector("#points").textContent = result.prediction.PTS.toFixed(1);
    document.querySelector("#rebounds").textContent = result.prediction.REB.toFixed(1);
    document.querySelector("#assists").textContent = result.prediction.AST.toFixed(1);
  } catch (error) {
    errorMessage.textContent = error.message || "The prediction could not be completed.";
  } finally {
    button.disabled = false;
    button.textContent = "Predict stats";
  }
});
