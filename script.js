document.getElementById('routeForm').addEventListener('submit', function(event) {
    const startLocation = document.getElementById('start_location').value;
    const destLocation = document.getElementById('dest_location').value;

    if (startLocation !== startLocation.toLowerCase() || destLocation !== destLocation.toLowerCase()) {
        event.preventDefault();
        alert('Please enter locations in lowercase letters only (e.g., "dwaraka nagar, visakhapatnam").');
    }
});