document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById("imageUpload");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Send the image to the backend
    const response = await fetch("/predict", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        alert("Error uploading the image.");
        return;
    }

    const result = await response.json();
    document.getElementById("result").style.display = "block";
    document.getElementById("prediction").innerText = `Predicted Class: ${result.predicted_class}`;

    const uploadedImage = document.getElementById("uploadedImage");
    uploadedImage.src = result.image_path;
    uploadedImage.alt = result.predicted_class;
});
