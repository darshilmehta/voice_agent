import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page page-state">
      <div className="empty-state">
        <strong className="empty-title">Page not found</strong>
        <div className="empty-text">
          <p>There's nothing at this address.</p>
        </div>
        <Link href="/" className="btn">
          Back to home
        </Link>
      </div>
    </div>
  );
}
