/* js/shop.js — PawCare Shop */

(function(){
  "use strict";

  const PRODUCTS = [
    { id:"bowl-bamboo",  title:"Bamboo Food Bowl",  price:599,  cat:"dining", img:"assets/products/bamboo-food-bowl.png",    desc:"Plant-based, dishwasher safe." },
    { id:"toy-rope",     title:"Hemp Rope Toy",     price:399,  cat:"toys",   img:"assets/products/hemp-rope.png",          desc:"Strong weave, compostable fibers." },
    { id:"shampoo-oats", title:"Oatmeal & Aloe Shampoo", price:549, cat:"care", img:"assets/products/shampoo-oat.png",      desc:"Soothing wash for sensitive skin." },
    { id:"bag-poop",     title:"Compostable Poop Bags (90)", price:329, cat:"waste", img:"assets/products/poop-bags.png",   desc:"Leak-proof, corn-starch based." },
    { id:"brush-bamboo", title:"Bamboo Grooming Brush", price:499, cat:"care", img:"assets/products/bamboo-brush.png",      desc:"Ergonomic handle, detangles gently." },
    { id:"snack-treats", title:"Pumpkin Crunch Treats", price:379, cat:"dining", img:"assets/products/pumpkin-treats.png",  desc:"Grain-free, oven-baked goodness." },
    { id:"bottle-steel", title:"Stainless Travel Bottle", price:899, cat:"walk", img:"assets/products/travel-bottle.png",   desc:"Double-wall bottle with bowl lid." },
    { id:"mat-natural",  title:"Natural Rubber Lick Mat", price:449, cat:"toys", img:"assets/products/lick-mat.png",         desc:"Slow-snack texture, BPA-free." },
    { id:"raincoat-bio", title:"Biodegradable Raincoat", price:999, cat:"wear", img:"assets/products/raincoat.png",          desc:"Lightweight, eco-conscious shell." },
    { id:"collar-led",   title:"LED Night Collar",  price:699,  cat:"wear", img:"assets/products/led-collar.png",           desc:"USB rechargeable, high-visibility." }
  ];

  const CART_KEY = "pawcare_cart";
  const INR = new Intl.NumberFormat("en-IN",{ style:"currency", currency:"INR", maximumFractionDigits:0 });

  const $ = (sel,ctx=document)=>ctx.querySelector(sel);

  function getCart(){ return JSON.parse(localStorage.getItem(CART_KEY)||"[]"); }
  function setCart(arr){ localStorage.setItem(CART_KEY, JSON.stringify(arr)); updateCartBadge(); }

  function addItem(id, qty=1){
    const cart = getCart();
    const line = cart.find(i=>i.id===id);
    if(line) line.qty += qty;
    else cart.push({id,qty});
    setCart(cart);
    renderCart();
  }

  function productById(id){ return PRODUCTS.find(p=>p.id===id); }
  function cartSubtotal(){ return getCart().reduce((s,i)=>{ const p=productById(i.id); return s+(p?p.price*i.qty:0); },0); }

  function updateCartBadge(){
    const n = getCart().reduce((s,i)=>s+i.qty,0);
    const el = $("#cartCount");
    if(el) el.textContent = n;
  }

  function refreshTotals(){
    const sub = cartSubtotal();
    const ship = localStorage.getItem("pawcare_is_golden")==="1" ? 0 : (sub>0?99:0);
    const total=sub+ship;
    $("#subTotal").textContent=INR.format(sub);
    $("#shipFee").textContent=INR.format(ship);
    $("#grandTotal").textContent=INR.format(total);
    $("#perkRow").hidden = !(localStorage.getItem("pawcare_is_golden")==="1");
    $("#goldenBadge").hidden = !(localStorage.getItem("pawcare_is_golden")==="1");
  }

  function renderGrid(){
    const grid=$("#grid"); grid.innerHTML="";
    const q=$("#q")?.value.toLowerCase()||"", cat=$("#cat")?.value||"";
    PRODUCTS.filter(p=>(!cat||p.cat===cat)&&(p.title.toLowerCase().includes(q)||p.desc.toLowerCase().includes(q)))
    .forEach(p=>{
      const tpl=$("#tplProductCard");
      let node=tpl.content.firstElementChild.cloneNode(true);
      $("img",node).src=p.img; $("img",node).alt=p.title;
      $(".title",node).textContent=p.title;
      $(".desc",node).textContent=p.desc;
      $(".price",node).textContent=INR.format(p.price);
      $(".btn-add",node).addEventListener("click",()=>addItem(p.id,1));
      grid.appendChild(node);
    });
  }

  function renderCart(){
    const list=$("#cartItems"); list.innerHTML="";
    const cart=getCart();
    if(cart.length===0){ list.innerHTML="<p>Your cart is empty.</p>"; refreshTotals(); return; }
    cart.forEach(it=>{
      const p=productById(it.id);
      list.innerHTML+=`<div>${p.title} × ${it.qty} — ${INR.format(p.price*it.qty)}</div>`;
    });
    refreshTotals();
  }

  document.addEventListener("DOMContentLoaded",()=>{
    renderGrid(); updateCartBadge(); refreshTotals();

    // Cart toggle
    const cartDrawer = $("#cartDrawer");
    const cartOverlay = $("#cartOverlay");
    const openCart = $("#openCart");
    const closeCart = $("#closeCart");

    function openDrawer() {
      cartDrawer.classList.add("open");
      cartDrawer.removeAttribute("aria-hidden");
      cartOverlay.hidden = false;
    }
    function closeDrawer() {
      cartDrawer.classList.remove("open");
      cartDrawer.setAttribute("aria-hidden", "true");
      cartOverlay.hidden = true;
    }

    if (openCart) openCart.addEventListener("click", openDrawer);
    if (closeCart) closeCart.addEventListener("click", closeDrawer);
    if (cartOverlay) cartOverlay.addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !cartOverlay.hidden) closeDrawer();
    });

    // Checkout
    $("#checkoutBtn").addEventListener("click",()=>{
      alert("Demo checkout");
      localStorage.removeItem(CART_KEY);
      renderCart();
    });

    // Filters
    $("#q").addEventListener("input",renderGrid);
    $("#cat").addEventListener("change",renderGrid);

    // Golden toggle
    const goldenToggle = $("#goldenToggle");
    if (goldenToggle){
      if (localStorage.getItem("pawcare_is_golden") === "1") {
        goldenToggle.checked = true;
      }
      goldenToggle.addEventListener("change", () => {
        localStorage.setItem("pawcare_is_golden", goldenToggle.checked ? "1" : "0");
        refreshTotals();
      });
    }
  });
})();
