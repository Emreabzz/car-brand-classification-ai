const form = document.getElementById("predictForm");
const output = document.getElementById("predictText");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);

  const res = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  let text = "🔍 Tahmin Sonuçları (Top-5)\n\n";
  data.forEach((item, i) => {
    text += `${i + 1}. ${item.class} — ${(item.confidence * 100).toFixed(2)}%\n`;
  });

  output.textContent = text;
});
