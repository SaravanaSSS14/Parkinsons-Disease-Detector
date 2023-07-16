// JavaScript source code



    // Resets the form on refresh
    window.addEventListener('load', function () {
            var form = document.getElementById('myForm');
    form.reset();
        });
    window.addEventListener('DOMContentLoaded', function () {
            var inputField = document.querySelector('input');
    inputField.classList.add('blink-caret');
    inputField.focus();
        });
    // Get all the input fields in the form
    const inputFields = document.querySelectorAll('input');

        // Add event listener to each input field
        inputFields.forEach((field, index) => {
        field.addEventListener('keydown', function (event) {
            if (event.key === 'Enter') {
                event.preventDefault(); // Prevent form submission

                const nextIndex = index + 1;
                if (nextIndex < inputFields.length) {
                    inputFields[nextIndex].focus(); // Move to the next field
                } else {
                    // Reached the last field, submit the form
                    document.getElementById('myForm').submit();
                    console.log('Form submitted!')
                }
            }
        });
        });

    const addressField = document.getElementById('address');
    addressField.addEventListener('input', function () {
        const content = this.value;
        const contentLength = content.length;

        // Calculate the font size based on content length
        const fontSize = Math.max(16 - Math.floor(contentLength / 10), 12);

        this.style.fontSize = `${fontSize}px`;
    });

    const textarea = document.getElementById("myTextarea");
    function handleKeyDown(event) {
        if (event.key === "Enter" && event.shiftKey) {
            const textarea = event.target;
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const value = textarea.value;
            textarea.value = value.substring(0, start) + "\n" + value.substring(end);
            textarea.selectionStart = textarea.selectionEnd = start + 1;
            event.preventDefault();
        }
}

//Form Validation
function validateForm() {
    // Perform form validation here
    var isValid = true;

    // Name validation
    var Name = document.getElementById("name").value;
    if (Name.trim() === "") {
        alert("Please enter your name");
        isValid = false;
    }

    // Gender validation
    var genderInputs = document.querySelectorAll('input[name="gender"]');
    var selectedGender = false;
    for (var i = 0; i < genderInputs.length; i++) {
        if (genderInputs[i].checked) {
            selectedGender = true;
            break;
        }
    }
    if (!selectedGender) {
        alert("Please select your gender");
        isValid = false;
    }

    // Age validation
    var age = document.getElementById("age").value;
    if (isNaN(age) || age < 1 || age > 120) {
        alert("Please enter a valid age between 1 and 120");
        isValid = false;
    }

    // Phone validation
    var phone = document.getElementById("phone").value;
    if (!/^\d{10}$/.test(phone)) {
        alert("Please enter a valid 10-digit phone number");
        isValid = false;
    }

    // Email validation
    var email = document.getElementById("email").value;
    if (!/^[\w-]+(\.[\w-]+)*@([\w-]+\.)+[a-zA-Z]{2,7}$/.test(email)) {
        alert("Please enter a valid email address");
        isValid = false;
    }

    if (isValid) {
        // Form is valid, proceed to the next page
        window.location.href = '../pages/upload.html';
        return false; // Prevent default form submission
    } else {
        // Form is invalid, display error message or perform actions
        alert('Please fill in all required fields correctly.'); // Example of displaying an error message
        return false; // Prevent default form submission
    }
}
