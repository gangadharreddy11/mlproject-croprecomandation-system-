document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("cropForm");

    form.addEventListener("submit", function (e) {
        // Simple front-end validation (optional)
        const inputs = form.querySelectorAll("input[type='number']");
        for (let input of inputs) {
            if (input.value === "") {
                alert("Please fill all fields.");
                e.preventDefault();
                return;
            }
        }
    });
});
