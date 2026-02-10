const form = document.getElementById("measureForm");
const imageInput = document.getElementById("imageInput");
const statusText = document.getElementById("statusText");
const inputPreview = document.getElementById("inputPreview");
const debugPreview = document.getElementById("debugPreview");
const inputFrame = document.getElementById("inputFrame");
const debugFrame = document.getElementById("debugFrame");
const jsonOutput = document.getElementById("jsonOutput");
const jsonLink = document.getElementById("jsonLink");

const setStatus = (text) => {
  statusText.textContent = text;
};

const showImage = (imgEl, frameEl, url) => {
  if (!url) return;
  imgEl.src = url;
  frameEl.classList.add("show");
  frameEl.querySelector(".placeholder").style.display = "none";
};

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    setStatus("Waiting for image…");
    return;
  }
  const url = URL.createObjectURL(file);
  showImage(inputPreview, inputFrame, url);
  setStatus("Image ready. Click to start measurement.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    setStatus("Please select an image first.");
    return;
  }

  setStatus("Measuring… Please wait.");
  jsonOutput.textContent = "{" + "\n  \"status\": \"processing\"\n}";

  const formData = new FormData(form);
  formData.append("image", file);

  try {
    const response = await fetch("/api/measure", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      setStatus(error.error || "Measurement failed");
      return;
    }

    const data = await response.json();

    jsonOutput.textContent = JSON.stringify(data.result, null, 2);
    jsonLink.href = data.result_json_url || "#";

    showImage(inputPreview, inputFrame, data.input_image_url);
    showImage(debugPreview, debugFrame, data.result_image_url);

    if (data.success) {
      setStatus("Measurement complete. Results updated.");
    } else {
      setStatus("Measurement failed. Check fail_reason.");
    }
  } catch (error) {
    setStatus("Network error. Please retry.");
  }
});
