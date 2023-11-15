document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const resultsDiv = document.getElementById('results');

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchInput.value;

        // Send the query to the server using fetch
        const response = await fetch(`/search?q=${query}`);
        const data = await response.json();

        // Display the results in the resultsDiv
        resultsDiv.innerHTML = data.result;
    });
});
