document.getElementById('promptBtn').addEventListener('click', async () => {
  const userInput = prompt("Enter something:");
  if (!userInput) return;

  const response = await fetch('/solve_tsp', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ input: userInput })
  });

  const data = await response.json();
  alert(data.result);
});
