const form = document.querySelector("#prediction-form");
const errorMessage = document.querySelector("#error");
const submitButton = form.querySelector("button");
const buttonLabel = document.querySelector("#button-label");
const resultContext = document.querySelector("#result-context");
const statElements = {
  PTS: document.querySelector("#pts"),
  REB: document.querySelector("#reb"),
  AST: document.querySelector("#ast"),
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.textContent = "";
  submitButton.disabled = true;
  buttonLabel.textContent = "Calculating…";

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

    for (const [stat, value] of Object.entries(result.prediction)) {
      statElements[stat].textContent = value.toFixed(1);
    }
    resultContext.textContent =
      `${result.player} · ${result.location === "home" ? "vs" : "at"} ${result.opponent}`;
  } catch (failure) {
    errorMessage.textContent = failure.message;
  } finally {
    submitButton.disabled = false;
    buttonLabel.textContent = "Run projection";
  }
});
