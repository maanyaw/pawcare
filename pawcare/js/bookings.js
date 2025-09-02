/* js/bookings.js
   PawCare Bookings integration
   - Sends booking data to Flask backend (/booking-apply)
   - Updates "Top Matches Near You" with demo providers
*/

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("bookingForm");
  const matchList = document.getElementById("matchList");
  const bufferAdvice = document.getElementById("bufferAdvice");
  const goldToggle = document.getElementById("goldBooking");

  if (!form) return; // safety

  // --- Handle form submission ---
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const data = Object.fromEntries(fd.entries());

    // Golden membership checkbox
    data.golden = goldToggle && goldToggle.checked ? 1 : 0;

    try {
      const res = await fetch("http://127.0.0.1:8000/booking-apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      if (!res.ok) throw new Error(`Server ${res.status}`);
      const out = await res.json();

      alert(out.msg || "✅ Booking submitted successfully!");
      updateMatches(data);
    } catch (err) {
      console.error("Booking error:", err);
      alert("❌ Error submitting booking: " + err.message);
    }
  });

  // --- Demo "Top Matches Near You" ---
  function updateMatches(profile) {
    // Hardcoded demo providers
    const providers = [
      { name: "Aisha", rating: 4.8, km: 2.1, exp: "dog:3", slot: "ok" },
      { name: "Rohit", rating: 4.6, km: 4.0, exp: "dog:2", slot: "alt" },
      { name: "Meera", rating: 4.9, km: 6.5, exp: "dog:3", slot: "ok" }
    ];

    if (matchList) {
      matchList.innerHTML = providers.map(p => `
        <li>
          <b>${p.name}</b> ★${p.rating} — ${p.km} km — ${p.exp} — ${p.slot}
        </li>
      `).join("");
    }

    if (bufferAdvice) {
      bufferAdvice.textContent = "Add a 10–15 min buffer during peak hours.";
    }
  }
});
